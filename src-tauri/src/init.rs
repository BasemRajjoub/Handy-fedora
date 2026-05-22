use crate::managers::audio::AudioRecordingManager;
use crate::managers::history::HistoryManager;
use crate::managers::model::ModelManager;
use crate::managers::transcription::TranscriptionManager;
use crate::transcription_coordinator::TranscriptionCoordinator;
use crate::{commands, portable, settings, signal_handle, tray, utils, FILE_LOG_LEVEL};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tauri::image::Image;
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Listener, Manager};
use tauri_plugin_autostart::ManagerExt;
use tracing_subscriber::{filter::FilterFn, fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

pub(crate) fn level_from_u8(value: u8) -> Option<tracing::Level> {
    match value {
        0 => None,
        1 => Some(tracing::Level::ERROR),
        2 => Some(tracing::Level::WARN),
        3 => Some(tracing::Level::INFO),
        4 => Some(tracing::Level::DEBUG),
        5 => Some(tracing::Level::TRACE),
        _ => Some(tracing::Level::TRACE),
    }
}

pub(crate) fn setup_tracing(log_dir: std::path::PathBuf) {
    // LogTracer::init() is called internally by .init() below (tracing-log feature).
    // Calling it here first causes a double-init panic. Removed.

    let console_filter = EnvFilter::try_from_env("HANDY_LOG")
        .unwrap_or_else(|_| EnvFilter::new("info"));
    let console_layer = fmt::layer().with_target(true).with_filter(console_filter);

    std::fs::create_dir_all(&log_dir).ok();
    let file_appender = tracing_appender::rolling::never(&log_dir, "handy.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
    Box::leak(Box::new(guard));
    let file_layer = fmt::layer()
        .with_ansi(false)
        .with_writer(non_blocking)
        .with_filter(FilterFn::new(|meta| {
            match level_from_u8(FILE_LOG_LEVEL.load(Ordering::Relaxed)) {
                None => false,
                Some(max) => *meta.level() <= max,
            }
        }));

    tracing_subscriber::registry()
        .with(console_layer)
        .with(file_layer)
        .init();
}

pub(crate) fn show_main_window(app: &AppHandle) {
    if let Some(main_window) = app.get_webview_window("main") {
        if let Err(e) = main_window.unminimize() {
            tracing::error!("Failed to unminimize webview window: {}", e);
        }
        if let Err(e) = main_window.show() {
            tracing::error!("Failed to show webview window: {}", e);
        }
        if let Err(e) = main_window.set_focus() {
            tracing::error!("Failed to focus webview window: {}", e);
        }
        #[cfg(target_os = "macos")]
        {
            if let Err(e) = app.set_activation_policy(tauri::ActivationPolicy::Regular) {
                tracing::error!("Failed to set activation policy to Regular: {}", e);
            }
        }
        return;
    }

    let webview_labels = app.webview_windows().keys().cloned().collect::<Vec<_>>();
    tracing::error!(
        "Main window not found. Webview labels: {:?}",
        webview_labels
    );
}

#[allow(unused_variables)]
pub(crate) fn should_force_show_permissions_window(app: &AppHandle) -> bool {
    #[cfg(target_os = "windows")]
    {
        let model_manager = app.state::<Arc<ModelManager>>();
        let has_downloaded_models = model_manager
            .get_available_models()
            .iter()
            .any(|model| model.is_downloaded);

        if !has_downloaded_models {
            return false;
        }

        let status = commands::audio::get_windows_microphone_permission_status();
        if status.supported && status.overall_access == commands::audio::PermissionAccess::Denied {
            tracing::info!(
                "Windows microphone permissions are denied; forcing main window visible for onboarding"
            );
            return true;
        }
    }

    false
}

pub(crate) fn initialize_core_logic(app_handle: &AppHandle) {
    let recording_manager = Arc::new(
        AudioRecordingManager::new(app_handle).expect("Failed to initialize recording manager"),
    );
    let model_manager =
        Arc::new(ModelManager::new(app_handle).expect("Failed to initialize model manager"));
    let transcription_manager = Arc::new(
        TranscriptionManager::new(app_handle, model_manager.clone())
            .expect("Failed to initialize transcription manager"),
    );
    let history_manager =
        Arc::new(HistoryManager::new(app_handle).expect("Failed to initialize history manager"));

    crate::managers::transcription::apply_accelerator_settings(app_handle);

    app_handle.manage(recording_manager.clone());
    app_handle.manage(model_manager.clone());
    app_handle.manage(transcription_manager.clone());
    app_handle.manage(history_manager.clone());

    #[cfg(unix)]
    {
        use signal_hook::consts::{SIGUSR1, SIGUSR2};
        use signal_hook::iterator::Signals;
        let signals = Signals::new(&[SIGUSR1, SIGUSR2])
            .expect("SIGUSR1/SIGUSR2 are valid signal numbers; registration must not fail");
        signal_handle::setup_signal_handler(app_handle.clone(), signals);
    }

    #[cfg(target_os = "macos")]
    {
        let settings = settings::get_settings(app_handle);
        if settings.start_hidden && settings.show_tray_icon {
            let _ = app_handle.set_activation_policy(tauri::ActivationPolicy::Accessory);
        }
    }

    let initial_theme = tray::get_current_theme(app_handle);
    let initial_icon_path = tray::get_icon_path(initial_theme, tray::TrayIconState::Idle);

    let tray = TrayIconBuilder::new()
        .icon(
            Image::from_path(
                app_handle
                    .path()
                    .resolve(initial_icon_path, tauri::path::BaseDirectory::Resource)
                    .unwrap(),
            )
            .unwrap(),
        )
        .tooltip(tray::tray_tooltip())
        .show_menu_on_left_click(true)
        .icon_as_template(true)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "settings" => {
                show_main_window(app);
            }
            "check_updates" => {
                let settings = settings::get_settings(app);
                if settings.update_checks_enabled {
                    show_main_window(app);
                    let _ = app.emit("check-for-updates", ());
                }
            }
            "copy_last_transcript" => {
                tray::copy_last_transcript(app);
            }
            "unload_model" => {
                let transcription_manager = app.state::<Arc<TranscriptionManager>>();
                if !transcription_manager.is_model_loaded() {
                    tracing::warn!("No model is currently loaded.");
                    return;
                }
                match transcription_manager.unload_model() {
                    Ok(()) => tracing::info!("Model unloaded via tray."),
                    Err(e) => tracing::error!("Failed to unload model via tray: {}", e),
                }
            }
            "cancel" => {
                use crate::utils::cancel_current_operation;
                cancel_current_operation(app);
            }
            "quit" => {
                app.exit(0);
            }
            id if id.starts_with("model_select:") => {
                let model_id = id.strip_prefix("model_select:").unwrap().to_string();
                let current_model = settings::get_settings(app).selected_model;
                if model_id == current_model {
                    return;
                }
                let app_clone = app.clone();
                std::thread::spawn(move || {
                    match commands::models::switch_active_model(&app_clone, &model_id) {
                        Ok(()) => {
                            tracing::info!("Model switched to {} via tray.", model_id);
                        }
                        Err(e) => {
                            tracing::error!("Failed to switch model via tray: {}", e);
                        }
                    }
                    tray::update_tray_menu(&app_clone, &tray::TrayIconState::Idle, None);
                });
            }
            _ => {}
        })
        .build(app_handle)
        .unwrap();
    app_handle.manage(tray);

    utils::update_tray_menu(app_handle, &utils::TrayIconState::Idle, None);

    let settings = settings::get_settings(app_handle);
    if !settings.show_tray_icon {
        tray::set_tray_visibility(app_handle, false);
    }

    let app_handle_for_listener = app_handle.clone();
    app_handle.listen("model-state-changed", move |_| {
        tray::update_tray_menu(&app_handle_for_listener, &tray::TrayIconState::Idle, None);
    });

    let autostart_manager = app_handle.autolaunch();
    let settings = settings::get_settings(app_handle);

    if settings.autostart_enabled {
        let _ = autostart_manager.enable();
    } else {
        let _ = autostart_manager.disable();
    }

    utils::create_recording_overlay(app_handle);
}
