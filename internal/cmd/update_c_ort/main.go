// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

var filesToDownload = map[string]string{
	"https://raw.githubusercontent.com/microsoft/onnxruntime/v1.26.0/include/onnxruntime/core/session/onnxruntime_c_api.h":    "internal/ort/onnxruntime_c_api.h",
	"https://raw.githubusercontent.com/microsoft/onnxruntime/v1.26.0/include/onnxruntime/core/session/onnxruntime_ep_c_api.h": "internal/ort/onnxruntime_ep_c_api.h",
}

func main() {
	for url, targetFile := range filesToDownload {
		fmt.Printf("Downloading %s...\n", url)
		resp, err := http.Get(url)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error fetching header: %v\n", err)
			os.Exit(1)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			fmt.Fprintf(os.Stderr, "Error: received status code %d\n", resp.StatusCode)
			os.Exit(1)
		}

		dir := filepath.Dir(targetFile)
		if err := os.MkdirAll(dir, 0755); err != nil {
			fmt.Fprintf(os.Stderr, "Error creating directory %s: %v\n", dir, err)
			os.Exit(1)
		}

		out, err := os.Create(targetFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating target file: %v\n", err)
			os.Exit(1)
		}
		defer out.Close()

		_, err = io.Copy(out, resp.Body)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error writing header file: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("Successfully downloaded and saved header to %s\n", targetFile)
	}
}
