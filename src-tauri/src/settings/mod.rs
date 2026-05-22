pub mod defaults;
pub mod types;

pub use defaults::{ensure_post_process_defaults, get_default_settings};
pub use types::*;

use tauri::AppHandle;
use tauri_plugin_store::StoreExt;
use tracing::{debug, warn};

pub const SETTINGS_STORE_PATH: &str = "settings_store.json";

pub fn load_or_create_app_settings(app: &AppHandle) -> AppSettings {
    let store = app
        .store(crate::portable::store_path(SETTINGS_STORE_PATH))
        .expect("Failed to initialize store");

    let mut settings = if let Some(settings_value) = store.get("settings") {
        match serde_json::from_value::<AppSettings>(settings_value) {
            Ok(mut settings) => {
                debug!("Found existing settings: {:?}", settings);
                let default_settings = get_default_settings();
                let mut updated = false;

                for (key, value) in default_settings.bindings {
                    if !settings.bindings.contains_key(&key) {
                        debug!("Adding missing binding: {}", key);
                        settings.bindings.insert(key, value);
                        updated = true;
                    }
                }

                if updated {
                    debug!("Settings updated with new bindings");
                    match serde_json::to_value(&settings) {
                        Ok(v) => { store.set("settings", v); }
                        Err(e) => tracing::error!("Failed to serialize settings: {e}"),
                    }
                }

                settings
            }
            Err(e) => {
                warn!("Failed to parse settings: {}", e);
                let default_settings = get_default_settings();
                match serde_json::to_value(&default_settings) {
                    Ok(v) => { store.set("settings", v); }
                    Err(e) => tracing::error!("Failed to serialize default settings: {e}"),
                }
                default_settings
            }
        }
    } else {
        let default_settings = get_default_settings();
        match serde_json::to_value(&default_settings) {
            Ok(v) => { store.set("settings", v); }
            Err(e) => tracing::error!("Failed to serialize default settings: {e}"),
        }
        default_settings
    };

    if ensure_post_process_defaults(&mut settings) {
        match serde_json::to_value(&settings) {
            Ok(v) => { store.set("settings", v); }
            Err(e) => tracing::error!("Failed to serialize settings: {e}"),
        }
    }

    settings
}

pub fn get_settings(app: &AppHandle) -> AppSettings {
    let store = app
        .store(crate::portable::store_path(SETTINGS_STORE_PATH))
        .expect("Failed to initialize store");

    let mut settings = if let Some(settings_value) = store.get("settings") {
        serde_json::from_value::<AppSettings>(settings_value).unwrap_or_else(|_| {
            let default_settings = get_default_settings();
            match serde_json::to_value(&default_settings) {
                Ok(v) => { store.set("settings", v); }
                Err(e) => tracing::error!("Failed to serialize default settings: {e}"),
            }
            default_settings
        })
    } else {
        let default_settings = get_default_settings();
        match serde_json::to_value(&default_settings) {
            Ok(v) => { store.set("settings", v); }
            Err(e) => tracing::error!("Failed to serialize default settings: {e}"),
        }
        default_settings
    };

    if ensure_post_process_defaults(&mut settings) {
        match serde_json::to_value(&settings) {
            Ok(v) => { store.set("settings", v); }
            Err(e) => tracing::error!("Failed to serialize settings: {e}"),
        }
    }

    settings
}

pub fn write_settings(app: &AppHandle, settings: AppSettings) {
    let store = app
        .store(crate::portable::store_path(SETTINGS_STORE_PATH))
        .expect("Failed to initialize store");

    match serde_json::to_value(&settings) {
        Ok(v) => { store.set("settings", v); }
        Err(e) => tracing::error!("Failed to serialize settings: {e}"),
    }
}

pub fn get_bindings(app: &AppHandle) -> std::collections::HashMap<String, ShortcutBinding> {
    get_settings(app).bindings
}

pub fn get_stored_binding(app: &AppHandle, id: &str) -> ShortcutBinding {
    let bindings = get_bindings(app);
    bindings
        .get(id)
        .unwrap_or_else(|| panic!("Binding '{id}' not found — caller must pass valid binding id"))
        .clone()
}

pub fn get_history_limit(app: &AppHandle) -> usize {
    get_settings(app).history_limit
}

pub fn get_recording_retention_period(app: &AppHandle) -> RecordingRetentionPeriod {
    get_settings(app).recording_retention_period
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_disable_auto_submit() {
        let settings = get_default_settings();
        assert!(!settings.auto_submit);
        assert_eq!(settings.auto_submit_key, AutoSubmitKey::Enter);
    }

    #[test]
    fn debug_output_redacts_api_keys() {
        let mut settings = get_default_settings();
        settings
            .post_process_api_keys
            .insert("openai".to_string(), "sk-proj-secret-key-12345".to_string());
        settings.post_process_api_keys.insert(
            "anthropic".to_string(),
            "sk-ant-secret-key-67890".to_string(),
        );
        settings
            .post_process_api_keys
            .insert("empty_provider".to_string(), "".to_string());

        let debug_output = format!("{:?}", settings);

        assert!(!debug_output.contains("sk-proj-secret-key-12345"));
        assert!(!debug_output.contains("sk-ant-secret-key-67890"));
        assert!(debug_output.contains("[REDACTED]"));
    }

    #[test]
    fn secret_map_debug_redacts_values() {
        let map = SecretMap(std::collections::HashMap::from([("key".into(), "secret".into())]));
        let out = format!("{:?}", map);
        assert!(!out.contains("secret"));
        assert!(out.contains("[REDACTED]"));
    }
}
