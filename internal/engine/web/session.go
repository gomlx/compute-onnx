// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build js && wasm

package web

import (
	"syscall/js"

	"github.com/pkg/errors"
)

// Session wraps a JavaScript ort.InferenceSession.
type Session struct {
	jsSession js.Value
}

// CreateSession creates an onnxruntime-web InferenceSession from model bytes.
func CreateSession(modelBytes []byte, executionProvider string) (*Session, error) {
	if err := EnsureORTLoaded(); err != nil {
		return nil, errors.Wrap(err, "failed to initialize onnxruntime-web")
	}

	global := js.Global()
	ortVal := global.Get("ort")
	if ortVal.IsUndefined() || ortVal.IsNull() {
		return nil, errors.New("window.ort is not available")
	}

	inferenceSession := ortVal.Get("InferenceSession")
	if inferenceSession.IsUndefined() || inferenceSession.IsNull() {
		return nil, errors.New("ort.InferenceSession is not available")
	}

	// Create Uint8Array for model bytes
	jsU8 := global.Get("Uint8Array").New(len(modelBytes))
	js.CopyBytesToJS(jsU8, modelBytes)

	// Create session options
	options := global.Get("Object").New()
	eps := global.Get("Array").New()
	if executionProvider != "" {
		eps.Call("push", executionProvider)
	} else {
		eps.Call("push", "wasm")
	}
	options.Set("executionProviders", eps)

	createPromise := inferenceSession.Call("create", jsU8, options)
	jsSess, err := Await(createPromise)
	if err != nil {
		return nil, errors.Wrap(err, "ort.InferenceSession.create failed")
	}

	return &Session{
		jsSession: jsSess,
	}, nil
}

// Run executes the inference session with input feeds map and returns the results map.
func (s *Session) Run(feeds js.Value) (js.Value, error) {
	if s.jsSession.IsUndefined() || s.jsSession.IsNull() {
		return js.Undefined(), errors.New("cannot run on finalized session")
	}

	runPromise := s.jsSession.Call("run", feeds)
	results, err := Await(runPromise)
	if err != nil {
		return js.Undefined(), errors.Wrap(err, "session.run failed")
	}
	return results, nil
}

// Destroy releases the session.
func (s *Session) Destroy() error {
	s.jsSession = js.Undefined()
	return nil
}
