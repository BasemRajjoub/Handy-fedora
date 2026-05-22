use crate::settings::{PasteMethod, TypingTool};
use crate::utils::{is_kde_wayland, is_wayland};
use std::process::Command;
use tracing::info;

pub(super) fn try_send_key_combo_linux(paste_method: &PasteMethod) -> Result<bool, String> {
    if is_wayland() {
        if !is_kde_wayland() && is_wtype_available() {
            info!("Using wtype for key combo");
            send_key_combo_via_wtype(paste_method)?;
            return Ok(true);
        }
        if is_dotool_available() {
            info!("Using dotool for key combo");
            send_key_combo_via_dotool(paste_method)?;
            return Ok(true);
        }
        if is_ydotool_available() {
            info!("Using ydotool for key combo");
            send_key_combo_via_ydotool(paste_method)?;
            return Ok(true);
        }
    } else {
        if is_xdotool_available() {
            info!("Using xdotool for key combo");
            send_key_combo_via_xdotool(paste_method)?;
            return Ok(true);
        }
        if is_ydotool_available() {
            info!("Using ydotool for key combo");
            send_key_combo_via_ydotool(paste_method)?;
            return Ok(true);
        }
    }
    Ok(false)
}

/// Returns `Ok(true)` if a native tool handled it, `Ok(false)` to fall back to enigo.
pub(super) fn try_direct_typing_linux(
    text: &str,
    preferred_tool: TypingTool,
) -> Result<bool, String> {
    if preferred_tool != TypingTool::Auto {
        return match preferred_tool {
            TypingTool::Wtype if is_wtype_available() => {
                info!("Using user-specified wtype");
                type_text_via_wtype(text)?;
                Ok(true)
            }
            TypingTool::Kwtype if is_kwtype_available() => {
                info!("Using user-specified kwtype");
                type_text_via_kwtype(text)?;
                Ok(true)
            }
            TypingTool::Dotool if is_dotool_available() => {
                info!("Using user-specified dotool");
                type_text_via_dotool(text)?;
                Ok(true)
            }
            TypingTool::Ydotool if is_ydotool_available() => {
                info!("Using user-specified ydotool");
                type_text_via_ydotool(text)?;
                Ok(true)
            }
            TypingTool::Xdotool if is_xdotool_available() => {
                info!("Using user-specified xdotool");
                type_text_via_xdotool(text)?;
                Ok(true)
            }
            _ => Err(format!(
                "Typing tool {:?} is not available on this system",
                preferred_tool
            )),
        };
    }

    if is_wayland() {
        if is_kde_wayland() && is_kwtype_available() {
            info!("Using kwtype for direct text input on KDE Wayland");
            type_text_via_kwtype(text)?;
            return Ok(true);
        }
        if !is_kde_wayland() && is_wtype_available() {
            info!("Using wtype for direct text input");
            type_text_via_wtype(text)?;
            return Ok(true);
        }
        if is_dotool_available() {
            info!("Using dotool for direct text input");
            type_text_via_dotool(text)?;
            return Ok(true);
        }
        if is_ydotool_available() {
            info!("Using ydotool for direct text input");
            type_text_via_ydotool(text)?;
            return Ok(true);
        }
    } else {
        if is_xdotool_available() {
            info!("Using xdotool for direct text input");
            type_text_via_xdotool(text)?;
            return Ok(true);
        }
        if is_ydotool_available() {
            info!("Using ydotool for direct text input");
            type_text_via_ydotool(text)?;
            return Ok(true);
        }
    }

    Ok(false)
}

/// Returns the list of available typing tools on this system.
/// Always includes "auto" as the first entry.
pub fn get_available_typing_tools() -> Vec<String> {
    let mut tools = vec!["auto".to_string()];
    if is_wtype_available()   { tools.push("wtype".to_string()); }
    if is_kwtype_available()  { tools.push("kwtype".to_string()); }
    if is_dotool_available()  { tools.push("dotool".to_string()); }
    if is_ydotool_available() { tools.push("ydotool".to_string()); }
    if is_xdotool_available() { tools.push("xdotool".to_string()); }
    tools
}

fn is_wtype_available() -> bool {
    Command::new("which").arg("wtype").output()
        .map(|o| o.status.success()).unwrap_or(false)
}

fn is_dotool_available() -> bool {
    Command::new("which").arg("dotool").output()
        .map(|o| o.status.success()).unwrap_or(false)
}

fn is_ydotool_available() -> bool {
    Command::new("which").arg("ydotool").output()
        .map(|o| o.status.success()).unwrap_or(false)
}

fn is_xdotool_available() -> bool {
    Command::new("which").arg("xdotool").output()
        .map(|o| o.status.success()).unwrap_or(false)
}

fn is_kwtype_available() -> bool {
    Command::new("which").arg("kwtype").output()
        .map(|o| o.status.success()).unwrap_or(false)
}

pub(super) fn is_wl_copy_available() -> bool {
    Command::new("which").arg("wl-copy").output()
        .map(|o| o.status.success()).unwrap_or(false)
}

fn type_text_via_wtype(text: &str) -> Result<(), String> {
    let output = Command::new("wtype").arg("--").arg(text).output()
        .map_err(|e| format!("Failed to execute wtype: {}", e))?;
    if !output.status.success() {
        return Err(format!("wtype failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(())
}

fn type_text_via_xdotool(text: &str) -> Result<(), String> {
    let output = Command::new("xdotool")
        .args(["type", "--clearmodifiers", "--"]).arg(text).output()
        .map_err(|e| format!("Failed to execute xdotool: {}", e))?;
    if !output.status.success() {
        return Err(format!("xdotool failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(())
}

fn type_text_via_dotool(text: &str) -> Result<(), String> {
    use std::io::Write;
    use std::process::Stdio;
    let mut child = Command::new("dotool").stdin(Stdio::piped()).spawn()
        .map_err(|e| format!("Failed to spawn dotool: {}", e))?;
    if let Some(mut stdin) = child.stdin.take() {
        writeln!(stdin, "type {}", text)
            .map_err(|e| format!("Failed to write to dotool stdin: {}", e))?;
    }
    let output = child.wait_with_output()
        .map_err(|e| format!("Failed to wait for dotool: {}", e))?;
    if !output.status.success() {
        return Err(format!("dotool failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(())
}

fn type_text_via_ydotool(text: &str) -> Result<(), String> {
    let output = Command::new("ydotool").args(["type", "--"]).arg(text).output()
        .map_err(|e| format!("Failed to execute ydotool: {}", e))?;
    if !output.status.success() {
        return Err(format!("ydotool failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(())
}

fn type_text_via_kwtype(text: &str) -> Result<(), String> {
    let output = Command::new("kwtype").arg("--").arg(text).output()
        .map_err(|e| format!("Failed to execute kwtype: {}", e))?;
    if !output.status.success() {
        return Err(format!("kwtype failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(())
}

/// Uses Stdio::null() to avoid blocking — wl-copy forks a daemon that inherits
/// piped fds, causing read_to_end to hang indefinitely.
pub(super) fn write_clipboard_via_wl_copy(text: &str) -> Result<(), String> {
    use std::process::Stdio;
    let status = Command::new("wl-copy").arg("--").arg(text)
        .stdout(Stdio::null()).stderr(Stdio::null()).status()
        .map_err(|e| format!("Failed to execute wl-copy: {}", e))?;
    if !status.success() {
        return Err("wl-copy failed".into());
    }
    Ok(())
}

fn send_key_combo_via_wtype(paste_method: &PasteMethod) -> Result<(), String> {
    let args: Vec<&str> = match paste_method {
        PasteMethod::CtrlV => vec!["-M", "ctrl", "-k", "v"],
        PasteMethod::ShiftInsert => vec!["-M", "shift", "-k", "Insert"],
        PasteMethod::CtrlShiftV => vec!["-M", "ctrl", "-M", "shift", "-k", "v"],
        _ => return Err("Unsupported paste method".into()),
    };
    let output = Command::new("wtype").args(&args).output()
        .map_err(|e| format!("Failed to execute wtype: {}", e))?;
    if !output.status.success() {
        return Err(format!("wtype failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(())
}

fn send_key_combo_via_dotool(paste_method: &PasteMethod) -> Result<(), String> {
    use std::process::Stdio;
    let command = match paste_method {
        PasteMethod::CtrlV => "echo key ctrl+v | dotool",
        PasteMethod::ShiftInsert => "echo key shift+insert | dotool",
        PasteMethod::CtrlShiftV => "echo key ctrl+shift+v | dotool",
        _ => return Err("Unsupported paste method".into()),
    };
    let status = Command::new("sh").args(["-c", command])
        .stdout(Stdio::null()).stderr(Stdio::null()).status()
        .map_err(|e| format!("Failed to execute dotool: {}", e))?;
    if !status.success() {
        return Err("dotool failed".into());
    }
    Ok(())
}

fn send_key_combo_via_ydotool(paste_method: &PasteMethod) -> Result<(), String> {
    let args: Vec<&str> = match paste_method {
        PasteMethod::CtrlV => vec!["key", "29:1", "47:1", "47:0", "29:0"],
        PasteMethod::ShiftInsert => vec!["key", "42:1", "110:1", "110:0", "42:0"],
        PasteMethod::CtrlShiftV => vec!["key", "29:1", "42:1", "47:1", "47:0", "42:0", "29:0"],
        _ => return Err("Unsupported paste method".into()),
    };
    let output = Command::new("ydotool").args(&args).output()
        .map_err(|e| format!("Failed to execute ydotool: {}", e))?;
    if !output.status.success() {
        return Err(format!("ydotool failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(())
}

fn send_key_combo_via_xdotool(paste_method: &PasteMethod) -> Result<(), String> {
    let key_combo = match paste_method {
        PasteMethod::CtrlV => "ctrl+v",
        PasteMethod::CtrlShiftV => "ctrl+shift+v",
        PasteMethod::ShiftInsert => "shift+Insert",
        _ => return Err("Unsupported paste method".into()),
    };
    let output = Command::new("xdotool").args(["key", "--clearmodifiers"]).arg(key_combo).output()
        .map_err(|e| format!("Failed to execute xdotool: {}", e))?;
    if !output.status.success() {
        return Err(format!("xdotool failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(())
}
