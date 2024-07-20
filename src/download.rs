use std::{
    fs::{create_dir_all, File},
    io::Write,
    path::PathBuf,
};

use burn::data::network::downloader;

/// Download the pre-trained weights to the local cache directory.
pub(crate) fn download(module: &'static str, url: &str) -> Result<PathBuf, std::io::Error> {
    // Model cache directory
    let model_dir = dirs::home_dir()
        .expect("Should be able to get home directory")
        .join(".cache")
        .join(format!("{}-burn", module));

    if !model_dir.exists() {
        create_dir_all(&model_dir)?;
    }

    let file_base_name = url.rsplit_once('/').unwrap().1;
    let file_name = model_dir.join(file_base_name);
    if !file_name.exists() {
        // Download file content
        let bytes = downloader::download_file_as_bytes(url, file_base_name);

        // Write content to file
        let mut output_file = File::create(&file_name)?;
        let bytes_written = output_file.write(&bytes)?;

        if bytes_written != bytes.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Failed to write the whole model weights file.",
            ));
        }
    }

    Ok(file_name)
}
