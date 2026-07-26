// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxruntime

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/pkg/errors"
)

const (
	DefaultVersion = ""
	RepoURL        = "https://github.com/microsoft/onnxruntime/releases/download"
)

// GetInstallPath returns the default directory where the library will be installed.
func GetInstallPath() (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", errors.Wrap(err, "failed to get user home directory")
	}
	switch runtime.GOOS {
	case "linux":
		return filepath.Join(homeDir, ".local", "lib", "onnxruntime"), nil
	case "darwin":
		return filepath.Join(homeDir, "Library", "Application Support", "onnxruntime"), nil
	case "windows":
		return filepath.Join(homeDir, "AppData", "Local", "onnxruntime"), nil
	default:
		return "", errors.Errorf("platform %s/%s not supported", runtime.GOOS, runtime.GOARCH)
	}
}

// GetLibFilename returns the expected library name for the platform.
func GetLibFilename() (string, error) {
	switch runtime.GOOS {
	case "linux":
		return "libonnxruntime.so", nil
	case "darwin":
		return "libonnxruntime.dylib", nil
	case "windows":
		return "onnxruntime.dll", nil
	default:
		return "", errors.Errorf("platform %s not supported", runtime.GOOS)
	}
}

// GetAssetURL returns the download URL for the given version, platform, and cuda setting.
func GetAssetURL(version string, cuda bool) (string, error) {
	var platform string
	var extension string

	switch runtime.GOOS {
	case "linux":
		extension = ".tgz"
		switch runtime.GOARCH {
		case "amd64":
			platform = "linux-x64"
		case "arm64":
			platform = "linux-aarch64"
		default:
			return "", errors.Errorf("unsupported linux arch: %s", runtime.GOARCH)
		}
	case "darwin":
		extension = ".tgz"
		switch runtime.GOARCH {
		case "arm64":
			platform = "osx-arm64"
		case "amd64":
			platform = "osx-x64"
		default:
			return "", errors.Errorf("unsupported osx arch: %s", runtime.GOARCH)
		}
	case "windows":
		extension = ".zip"
		switch runtime.GOARCH {
		case "amd64":
			platform = "win-x64"
		case "arm64":
			platform = "win-arm64"
		default:
			return "", errors.Errorf("unsupported windows arch: %s", runtime.GOARCH)
		}
	default:
		return "", errors.Errorf("unsupported OS: %s", runtime.GOOS)
	}

	if cuda {
		if runtime.GOARCH != "amd64" {
			return "", errors.Errorf("CUDA is only supported on amd64/x64 architecture (got %s)", runtime.GOARCH)
		}
		if runtime.GOOS != "linux" && runtime.GOOS != "windows" {
			return "", errors.Errorf("CUDA is only supported on linux or windows OS (got %s)", runtime.GOOS)
		}

		// List of candidate suffixes for CUDA flavors to try
		candidates := []string{
			"gpu_cuda12",
			"gpu",
			"gpu_cuda13",
		}

		for _, suffix := range candidates {
			candidatePlatform := fmt.Sprintf("%s-%s", platform, suffix)
			archiveName := fmt.Sprintf("onnxruntime-%s-%s%s", candidatePlatform, version, extension)
			url := fmt.Sprintf("%s/v%s/%s", RepoURL, version, archiveName)

			// Check if URL exists
			resp, err := http.Head(url)
			if err == nil {
				resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					return url, nil
				}
			}
		}

		return "", errors.Errorf("failed to find a valid CUDA package URL for version %s on platform %s", version, platform)
	}

	archiveName := fmt.Sprintf("onnxruntime-%s-%s%s", platform, version, extension)
	return fmt.Sprintf("%s/v%s/%s", RepoURL, version, archiveName), nil
}

// GetLatestVersion fetches the latest release version of ONNX Runtime from GitHub.
func GetLatestVersion() (string, error) {
	resp, err := http.Head("https://github.com/microsoft/onnxruntime/releases/latest")
	if err != nil {
		return "", errors.Wrap(err, "failed to fetch latest onnxruntime release from GitHub")
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 400 {
		return "", errors.Errorf("failed to fetch latest onnxruntime release from GitHub: status code %d", resp.StatusCode)
	}

	if resp.Request == nil {
		return "", errors.New("failed to fetch latest onnxruntime release: response Request is nil")
	}

	finalURL := resp.Request.URL.String()
	// The URL is expected to be of the form: https://github.com/microsoft/onnxruntime/releases/tag/vX.Y.Z
	// We want to extract "X.Y.Z"
	parts := strings.Split(finalURL, "/tag/")
	if len(parts) != 2 {
		return "", errors.Errorf("unexpected redirect URL when fetching latest release: %s", finalURL)
	}

	version := strings.TrimPrefix(parts[1], "v")
	return version, nil
}

// Install downloads and installs the ONNX Runtime library if not already present.
// It returns the absolute path to the main shared library file.
func Install(version string, cuda bool, force bool) (string, error) {
	if version == "" {
		if cuda {
			version = "1.26.0"
		} else {
			version = DefaultVersion
		}
	}
	if version == "" {
		var err error
		version, err = GetLatestVersion()
		if err != nil {
			return "", err
		}
	}

	installDir, err := GetInstallPath()
	if err != nil {
		return "", err
	}

	libFilename, err := GetLibFilename()
	if err != nil {
		return "", err
	}

	targetPath := filepath.Join(installDir, libFilename)
	if !force {
		if _, err := os.Stat(targetPath); err == nil {
			if !cuda {
				return targetPath, nil
			}
			cudaLibPath := filepath.Join(installDir, "libonnxruntime_providers_cuda.so")
			if _, err := os.Stat(cudaLibPath); err == nil {
				return targetPath, nil
			}
		}
	}

	// Needs installation
	url, err := GetAssetURL(version, cuda)
	if err != nil {
		return "", err
	}

	fmt.Printf("Installing ONNX Runtime %s to %s...\n", version, installDir)
	fmt.Printf("Downloading %s ...\n", url)

	resp, err := http.Get(url)
	if err != nil {
		return "", errors.Wrapf(err, "failed to download from %s", url)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", errors.Errorf("failed to download from %s: HTTP %s", url, resp.Status)
	}

	// Create temporary directory for extraction
	if err := os.MkdirAll(installDir, 0755); err != nil {
		return "", errors.Wrapf(err, "failed to create directory %s", installDir)
	}

	tmpDir, err := os.MkdirTemp(installDir, ".extracting-")
	if err != nil {
		return "", errors.Wrap(err, "failed to create temp extraction directory")
	}
	defer os.RemoveAll(tmpDir)

	if strings.HasSuffix(url, ".tgz") {
		err = extractTarGz(resp.Body, tmpDir, libFilename)
	} else if strings.HasSuffix(url, ".zip") {
		err = extractZip(resp.Body, resp.ContentLength, tmpDir, libFilename)
	} else {
		err = errors.Errorf("unknown archive format for %s", url)
	}
	if err != nil {
		return "", err
	}

	// Move the extracted files into place
	files, err := os.ReadDir(tmpDir)
	if err != nil {
		return "", errors.Wrap(err, "failed to read temp extraction directory")
	}

	if len(files) == 0 {
		return "", errors.Errorf("no library files extracted from the archive")
	}

	for _, file := range files {
		src := filepath.Join(tmpDir, file.Name())
		dest := filepath.Join(installDir, file.Name())

		// Remove existing dest if it's there
		_ = os.Remove(dest)
		if err := os.Rename(src, dest); err != nil {
			return "", errors.Wrapf(err, "failed to move extracted file %s to %s", file.Name(), dest)
		}
		fmt.Printf("Installed %s\n", dest)
	}

	return targetPath, nil
}

func extractTarGz(r io.Reader, destDir, libFilename string) error {
	gzr, err := gzip.NewReader(r)
	if err != nil {
		return errors.Wrap(err, "failed to create gzip reader")
	}
	defer gzr.Close()

	tr := tar.NewReader(gzr)
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return errors.Wrap(err, "failed to read tar header")
		}

		baseName := filepath.Base(header.Name)
		// We only want the library files, which are in the 'lib/' folder
		// Typically they contain 'libonnxruntime' or the specific libFilename.
		if header.Typeflag == tar.TypeReg || header.Typeflag == tar.TypeSymlink {
			if strings.Contains(header.Name, "/lib/") && (strings.Contains(baseName, "libonnxruntime") || baseName == libFilename) {
				targetFile := filepath.Join(destDir, baseName)
				if header.Typeflag == tar.TypeSymlink {
					// We resolve/write symlinks, or write them as symlink.
					// For Go loads, symlinks are fine or copy the target directly.
					// Let's create the symlink.
					_ = os.Remove(targetFile)
					err = os.Symlink(header.Linkname, targetFile)
					if err != nil {
						return errors.Wrapf(err, "failed to create symlink %s -> %s", targetFile, header.Linkname)
					}
				} else {
					f, err := os.OpenFile(targetFile, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0755)
					if err != nil {
						return errors.Wrapf(err, "failed to create file %s", targetFile)
					}
					if _, err := io.Copy(f, tr); err != nil {
						f.Close()
						return errors.Wrapf(err, "failed to write file %s", targetFile)
					}
					f.Close()
				}
			}
		}
	}
	return nil
}

func extractZip(r io.Reader, contentLength int64, destDir, libFilename string) error {
	// zip.NewReader requires ReaderAt, so we must write to a temp file first.
	tmpZipFile, err := os.CreateTemp("", "onnxruntime-*.zip")
	if err != nil {
		return errors.Wrap(err, "failed to create temporary zip file")
	}
	defer func() {
		tmpZipFile.Close()
		os.Remove(tmpZipFile.Name())
	}()

	if _, err := io.Copy(tmpZipFile, r); err != nil {
		return errors.Wrap(err, "failed to save archive to temporary file")
	}

	stat, err := tmpZipFile.Stat()
	if err != nil {
		return errors.Wrap(err, "failed to stat temporary zip file")
	}

	zr, err := zip.NewReader(tmpZipFile, stat.Size())
	if err != nil {
		return errors.Wrap(err, "failed to create zip reader")
	}

	for _, file := range zr.File {
		baseName := filepath.Base(file.Name)
		if !file.FileInfo().IsDir() && (strings.Contains(file.Name, "/lib/") || strings.Contains(file.Name, "/bin/")) &&
			(strings.Contains(baseName, "onnxruntime") || baseName == libFilename) {
			targetFile := filepath.Join(destDir, baseName)
			f, err := os.OpenFile(targetFile, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0755)
			if err != nil {
				return errors.Wrapf(err, "failed to create file %s", targetFile)
			}
			src, err := file.Open()
			if err != nil {
				f.Close()
				return errors.Wrapf(err, "failed to open zip file entry %s", file.Name)
			}
			_, err = io.Copy(f, src)
			src.Close()
			f.Close()
			if err != nil {
				return errors.Wrapf(err, "failed to write file %s", targetFile)
			}
		}
	}
	return nil
}
