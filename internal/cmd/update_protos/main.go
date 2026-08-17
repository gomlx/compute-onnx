// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

const (
	onnxGitHubRawUrl = "https://raw.githubusercontent.com/onnx/onnx/main/onnx/"
	goProtosPackage  = "github.com/gomlx/compute-onnx/support/protos"
)

var protoFiles = []string{
	"onnx-ml.proto",
	"onnx-operators-ml.proto",
	"onnx-data.proto",
}

func must(err error) {
	if err != nil {
		log.Fatalf("Error:\n%+v\n", err)
	}
}

func main() {
	cwd, err := os.Getwd()
	must(err)

	var targetDir string
	if filepath.Base(cwd) == "protos" && filepath.Base(filepath.Dir(cwd)) == "support" {
		targetDir = cwd
	} else {
		targetDir = filepath.Join(cwd, "support", "protos")
	}

	must(os.MkdirAll(targetDir, 0755))
	fmt.Printf("Writing protos and generated Go code to: %s\n", targetDir)

	for _, protoFile := range protoFiles {
		url := onnxGitHubRawUrl + protoFile + "3"
		fmt.Printf("Downloading %s ...\n", url)
		resp, err := http.Get(url)
		must(err)
		if resp.StatusCode != http.StatusOK {
			must(fmt.Errorf("failed to download %s: %s", url, resp.Status))
		}
		defer resp.Body.Close()

		content, err := io.ReadAll(resp.Body)
		must(err)

		content = removeGoPackageOption(content)
		content = fixPackageName(content)
		content = fixImports(content)

		destPath := filepath.Join(targetDir, protoFile)
		must(os.WriteFile(destPath, content, 0644))
		fmt.Printf("Wrote updated proto to %s\n", destPath)
	}

	goOpts := make([]string, len(protoFiles))
	for ii, protoFile := range protoFiles {
		goOpts[ii] = fmt.Sprintf("--go_opt=M%s=%s", protoFile, goProtosPackage)
	}

	for _, protoFile := range protoFiles {
		args := []string{
			"--go_out=" + targetDir,
			"-I=" + targetDir,
			fmt.Sprintf("--go_opt=module=%s", goProtosPackage),
		}
		args = append(args, goOpts...)
		args = append(args, filepath.Join(targetDir, protoFile))

		fmt.Printf("Running: protoc %s\n", strings.Join(args, " "))
		cmd := exec.Command("protoc", args...)
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		cmd.Stdout = os.Stdout

		if err := cmd.Run(); err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "Error executing protoc for %s: %v\n", protoFile, err)
			_, _ = fmt.Fprintf(os.Stderr, "Stderr:\n%s\n", stderr.String())
			os.Exit(1)
		}
		fmt.Printf("Generated Go wrappers for %s\n", protoFile)
	}
}

var reRemoveGoPackageOption = regexp.MustCompile(`option\s+go_package\s*=\s*"[^"]*?";`)

func removeGoPackageOption(content []byte) []byte {
	return reRemoveGoPackageOption.ReplaceAll(content, []byte{})
}

var rePackageName = regexp.MustCompile(`package onnx;`)

func fixPackageName(content []byte) []byte {
	return rePackageName.ReplaceAll(content, []byte(`package protos;`))
}

var reImports = regexp.MustCompile(`import\s+"onnx/(.*)3";`)

func fixImports(content []byte) []byte {
	return reImports.ReplaceAll(content, []byte(`import "$1";`))
}
