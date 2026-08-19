// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

const (
	defaultVersion = "1.22.0"
	cdnBaseURL     = "https://cdn.jsdelivr.net/npm/onnxruntime-web"
)

var webFiles = []string{
	"ort.min.js",
	"ort.min.js.map",
	"ort-wasm-simd-threaded.wasm",
	"ort-wasm-simd-threaded.mjs",
	"ort-wasm-simd-threaded.jsep.wasm",
	"ort-wasm-simd-threaded.jsep.mjs",
}

func main() {
	versionFlag := flag.String("version", defaultVersion, "onnxruntime-web version to download")
	outputDirFlag := flag.String("dir", "dist", "destination directory for downloaded web assets")
	flag.Parse()

	if err := os.MkdirAll(*outputDirFlag, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output directory %q: %v\n", *outputDirFlag, err)
		os.Exit(1)
	}

	fmt.Printf("Downloading onnxruntime-web v%s files into %s...\n", *versionFlag, *outputDirFlag)
	for _, filename := range webFiles {
		url := fmt.Sprintf("%s@%s/dist/%s", cdnBaseURL, *versionFlag, filename)
		destPath := filepath.Join(*outputDirFlag, filename)
		fmt.Printf("Fetching %s -> %s\n", url, destPath)

		resp, err := http.Get(url)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error fetching %s: %v\n", url, err)
			os.Exit(1)
		}
		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			fmt.Fprintf(os.Stderr, "Error: received HTTP %d for %s\n", resp.StatusCode, url)
			os.Exit(1)
		}

		out, err := os.Create(destPath)
		if err != nil {
			resp.Body.Close()
			fmt.Fprintf(os.Stderr, "Error creating file %s: %v\n", destPath, err)
			os.Exit(1)
		}

		_, err = io.Copy(out, resp.Body)
		resp.Body.Close()
		out.Close()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error writing file %s: %v\n", destPath, err)
			os.Exit(1)
		}
	}

	fmt.Println("Successfully downloaded onnxruntime-web assets.")
}
