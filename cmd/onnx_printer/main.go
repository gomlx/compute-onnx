// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package main provides the onnx_printer tool to pretty-print ONNX model files.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"
	"unsafe"

	"github.com/gomlx/compute-onnx/internal/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"google.golang.org/protobuf/proto"
)

var (
	maxItems int
	showDoc  bool
)

func init() {
	flag.IntVar(&maxItems, "max_items", 10, "Maximum number of items/numbers to print for tensor constants/initializers.")
	flag.IntVar(&maxItems, "n", 10, "Alias for -max_items.")
	flag.BoolVar(&showDoc, "show_doc", false, "Include docstrings in output if present.")
}

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <path-to-model.onnx>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "       cat model.onnx | %s [flags]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Pretty-prints the contents of an ONNX model file.\n\nFlags:\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	var reader io.Reader
	filePath := ""

	if flag.NArg() > 0 && flag.Arg(0) != "-" {
		filePath = flag.Arg(0)
		f, err := os.Open(filePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error opening file %q: %v\n", filePath, err)
			os.Exit(1)
		}
		defer f.Close()
		reader = f
	} else {
		filePath = "<stdin>"
		reader = os.Stdin
	}

	data, err := io.ReadAll(reader)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading ONNX data: %v\n", err)
		os.Exit(1)
	}

	if len(data) == 0 {
		fmt.Fprintf(os.Stderr, "Error: empty ONNX input data\n")
		os.Exit(1)
	}

	model := &protos.ModelProto{}
	if err := proto.Unmarshal(data, model); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing ONNX model proto: %v\n", err)
		os.Exit(1)
	}

	PrintModel(os.Stdout, model, filePath, maxItems, showDoc)
}

// PrintModel pretty-prints an ONNX ModelProto to the given writer.
func PrintModel(w io.Writer, model *protos.ModelProto, filename string, maxItems int, showDoc bool) {
	fmt.Fprintf(w, "ONNX Model: %s\n", filename)
	fmt.Fprintln(w, strings.Repeat("=", 80))
	fmt.Fprintf(w, "IR Version:       %d\n", model.GetIrVersion())
	if model.GetProducerName() != "" {
		prod := model.GetProducerName()
		if model.GetProducerVersion() != "" {
			prod += " (version " + model.GetProducerVersion() + ")"
		}
		fmt.Fprintf(w, "Producer:         %s\n", prod)
	}
	if model.GetDomain() != "" {
		fmt.Fprintf(w, "Domain:           %s\n", model.GetDomain())
	}
	if model.GetModelVersion() != 0 {
		fmt.Fprintf(w, "Model Version:    %d\n", model.GetModelVersion())
	}
	if showDoc && model.GetDocString() != "" {
		fmt.Fprintf(w, "Docstring:        %s\n", model.GetDocString())
	}
	if len(model.GetOpsetImport()) > 0 {
		fmt.Fprintln(w, "Opset Imports:")
		for _, op := range model.GetOpsetImport() {
			dom := op.GetDomain()
			if dom == "" {
				dom = "ai.onnx"
			}
			fmt.Fprintf(w, "  - %s (v%d)\n", dom, op.GetVersion())
		}
	}
	if len(model.GetMetadataProps()) > 0 {
		fmt.Fprintln(w, "Metadata:")
		for _, prop := range model.GetMetadataProps() {
			fmt.Fprintf(w, "  - %s: %s\n", prop.GetKey(), prop.GetValue())
		}
	}

	if g := model.GetGraph(); g != nil {
		fmt.Fprintln(w)
		PrintGraph(w, g, "", maxItems, showDoc)
	}

	if len(model.GetFunctions()) > 0 {
		fmt.Fprintln(w)
		fmt.Fprintf(w, "Functions (%d):\n", len(model.GetFunctions()))
		fmt.Fprintln(w, strings.Repeat("-", 80))
		for i, fn := range model.GetFunctions() {
			fmt.Fprintf(w, "[%d] Function: %s (domain: %s)\n", i, fn.GetName(), fn.GetDomain())
			if len(fn.GetInput()) > 0 {
				fmt.Fprintf(w, "    Inputs:  %s\n", strings.Join(fn.GetInput(), ", "))
			}
			if len(fn.GetOutput()) > 0 {
				fmt.Fprintf(w, "    Outputs: %s\n", strings.Join(fn.GetOutput(), ", "))
			}
			if len(fn.GetNode()) > 0 {
				fmt.Fprintf(w, "    Nodes (%d):\n", len(fn.GetNode()))
				shapeMap := make(map[string]shapes.Shape)
				for j, n := range fn.GetNode() {
					printNode(w, n, j, "      ", shapeMap, maxItems, showDoc)
				}
			}
		}
	}
}

// PrintGraph pretty-prints an ONNX GraphProto.
func PrintGraph(w io.Writer, g *protos.GraphProto, indent string, maxItems int, showDoc bool) {
	name := g.GetName()
	if name == "" {
		name = "<unnamed>"
	}
	fmt.Fprintf(w, "%sGraph: %s\n", indent, name)
	fmt.Fprintf(w, "%s%s\n", indent, strings.Repeat("=", 80-len(indent)))
	if showDoc && g.GetDocString() != "" {
		fmt.Fprintf(w, "%sDocstring: %s\n", indent, g.GetDocString())
	}

	shapeMap := make(map[string]shapes.Shape)

	// Collect shapes from Inputs, Outputs, ValueInfo, Initializers, and Constants
	for _, in := range g.GetInput() {
		if sh, ok := ShapeFromONNXType(in.GetType()); ok {
			shapeMap[in.GetName()] = sh
		}
	}
	for _, out := range g.GetOutput() {
		if sh, ok := ShapeFromONNXType(out.GetType()); ok {
			shapeMap[out.GetName()] = sh
		}
	}
	for _, vi := range g.GetValueInfo() {
		if sh, ok := ShapeFromONNXType(vi.GetType()); ok {
			shapeMap[vi.GetName()] = sh
		}
	}
	for _, init := range g.GetInitializer() {
		sh := ShapeFromTensorProto(init)
		if sh.Ok() {
			shapeMap[init.GetName()] = sh
		}
	}
	for _, n := range g.GetNode() {
		if n.GetOpType() == "Constant" {
			for _, attr := range n.GetAttribute() {
				if attr.GetName() == "value" && attr.GetT() != nil {
					sh := ShapeFromTensorProto(attr.GetT())
					for _, outName := range n.GetOutput() {
						if sh.Ok() {
							shapeMap[outName] = sh
						}
					}
				}
			}
		}
	}

	// Inputs
	if len(g.GetInput()) > 0 {
		fmt.Fprintf(w, "%sInputs (%d):\n", indent, len(g.GetInput()))
		for _, in := range g.GetInput() {
			doc := ""
			if showDoc && in.GetDocString() != "" {
				doc = fmt.Sprintf(" (%s)", in.GetDocString())
			}
			fmt.Fprintf(w, "%s  - %s%s\n", indent, formatValueWithShapeHeader(in.GetName(), shapeMap), doc)
		}
	}

	// Outputs
	if len(g.GetOutput()) > 0 {
		fmt.Fprintf(w, "%sOutputs (%d):\n", indent, len(g.GetOutput()))
		for _, out := range g.GetOutput() {
			doc := ""
			if showDoc && out.GetDocString() != "" {
				doc = fmt.Sprintf(" (%s)", out.GetDocString())
			}
			fmt.Fprintf(w, "%s  - %s%s\n", indent, formatValueWithShapeHeader(out.GetName(), shapeMap), doc)
		}
	}

	// Initializers (Constants)
	if len(g.GetInitializer()) > 0 {
		fmt.Fprintf(w, "%sInitializers (%d):\n", indent, len(g.GetInitializer()))
		for _, init := range g.GetInitializer() {
			sh := ShapeFromTensorProto(init)
			valStr := FormatTensorValues(init, maxItems)
			fmt.Fprintf(w, "%s  - %s: %s = %s\n", indent, init.GetName(), sh.String(), valStr)
		}
	}

	// Value Info (intermediate shapes)
	if len(g.GetValueInfo()) > 0 {
		fmt.Fprintf(w, "%sValue Info (%d):\n", indent, len(g.GetValueInfo()))
		for _, vi := range g.GetValueInfo() {
			fmt.Fprintf(w, "%s  - %s\n", indent, formatValueWithShapeHeader(vi.GetName(), shapeMap))
		}
	}

	// Nodes
	if len(g.GetNode()) > 0 {
		fmt.Fprintf(w, "%sNodes (%d):\n", indent, len(g.GetNode()))
		for i, n := range g.GetNode() {
			printNode(w, n, i, indent+"  ", shapeMap, maxItems, showDoc)
		}
	}
}

func printNode(w io.Writer, n *protos.NodeProto, index int, indent string, shapeMap map[string]shapes.Shape, maxItems int, showDoc bool) {
	// 1. Outputs string
	outs := n.GetOutput()
	var outStr string
	if len(outs) == 1 {
		outStr = outs[0]
	} else if len(outs) > 1 {
		outStr = "(" + strings.Join(outs, ", ") + ")"
	} else {
		outStr = "_"
	}

	// 2. OpType
	op := n.GetOpType()
	if n.GetDomain() != "" && n.GetDomain() != "ai.onnx" {
		op = n.GetDomain() + "." + op
	}

	// 3. Inputs string
	inStrs := make([]string, len(n.GetInput()))
	for i, inName := range n.GetInput() {
		inStrs[i] = formatValueWithShape(inName, shapeMap)
	}
	inputsFormatted := strings.Join(inStrs, ", ")

	// 4. Attributes string
	attrStr := ""
	if len(n.GetAttribute()) > 0 {
		attrParts := make([]string, len(n.GetAttribute()))
		for i, attr := range n.GetAttribute() {
			attrParts[i] = formatAttribute(attr, maxItems)
		}
		attrStr = " {" + strings.Join(attrParts, ", ") + "}"
	}

	// 5. Output Shapes string
	var outShapes []string
	for _, outName := range outs {
		if sh, ok := shapeMap[outName]; ok && sh.Ok() {
			outShapes = append(outShapes, sh.String())
		}
	}
	outShapesStr := ""
	if len(outShapes) > 0 {
		outShapesStr = " : " + strings.Join(outShapes, ", ")
	}

	// Single line format for op
	fmt.Fprintf(w, "%s[%d] %s: %s(%s)%s%s\n", indent, index, outStr, op, inputsFormatted, attrStr, outShapesStr)

	if showDoc && n.GetDocString() != "" {
		fmt.Fprintf(w, "%s    Doc: %s\n", indent, n.GetDocString())
	}

	// Sub-graphs if any attribute contains a graph
	for _, attr := range n.GetAttribute() {
		if attr.GetType() == protos.AttributeProto_GRAPH && attr.GetG() != nil {
			fmt.Fprintf(w, "%s    Sub-graph for %q:\n", indent, attr.GetName())
			PrintGraph(w, attr.GetG(), indent+"      ", maxItems, showDoc)
		}
	}
}

func formatValueWithShape(name string, shapeMap map[string]shapes.Shape) string {
	if name == "" {
		return "<empty>"
	}
	if sh, ok := shapeMap[name]; ok && sh.Ok() {
		return fmt.Sprintf("%s:%s", name, sh.String())
	}
	return name
}

func formatValueWithShapeHeader(name string, shapeMap map[string]shapes.Shape) string {
	if name == "" {
		return "<empty>"
	}
	if sh, ok := shapeMap[name]; ok && sh.Ok() {
		return fmt.Sprintf("%s: %s", name, sh.String())
	}
	return name
}

// ShapeFromONNXType converts an ONNX TypeProto (tensor type) into a GoMLX shapes.Shape.
func ShapeFromONNXType(t *protos.TypeProto) (shapes.Shape, bool) {
	if t == nil {
		return shapes.Invalid(), false
	}
	tt := t.GetTensorType()
	if tt == nil {
		return shapes.Invalid(), false
	}
	dt := DTypeFromONNX(protos.TensorProto_DataType(tt.GetElemType()))
	shapeProto := tt.GetShape()
	if shapeProto == nil {
		return shapes.Make(dt), true
	}
	dimsProto := shapeProto.GetDim()
	if len(dimsProto) == 0 {
		return shapes.Make(dt), true
	}

	dims := make([]int, len(dimsProto))
	axisNames := make([]string, len(dimsProto))
	hasDynamic := false
	hasAxisNames := false

	for i, d := range dimsProto {
		switch v := d.GetValue().(type) {
		case *protos.TensorShapeProto_Dimension_DimValue:
			dims[i] = int(v.DimValue)
			axisNames[i] = ""
		case *protos.TensorShapeProto_Dimension_DimParam:
			dims[i] = shapes.DynamicDim
			axisNames[i] = v.DimParam
			hasDynamic = true
			if v.DimParam != "" {
				hasAxisNames = true
			}
		default:
			dims[i] = shapes.DynamicDim
			axisNames[i] = "?"
			hasDynamic = true
			hasAxisNames = true
		}
	}

	if hasDynamic || hasAxisNames {
		return shapes.MakeDynamic(dt, dims, axisNames), true
	}
	return shapes.Make(dt, dims...), true
}

// DTypeFromONNX converts an ONNX TensorProto_DataType enum value into a GoMLX dtypes.DType.
func DTypeFromONNX(dt protos.TensorProto_DataType) dtypes.DType {
	switch dt {
	case protos.TensorProto_FLOAT:
		return dtypes.Float32
	case protos.TensorProto_DOUBLE:
		return dtypes.Float64
	case protos.TensorProto_FLOAT16:
		return dtypes.Float16
	case protos.TensorProto_BFLOAT16:
		return dtypes.BFloat16
	case protos.TensorProto_INT32:
		return dtypes.Int32
	case protos.TensorProto_INT64:
		return dtypes.Int64
	case protos.TensorProto_BOOL:
		return dtypes.Bool
	case protos.TensorProto_INT8:
		return dtypes.Int8
	case protos.TensorProto_UINT8:
		return dtypes.Uint8
	case protos.TensorProto_INT16:
		return dtypes.Int16
	case protos.TensorProto_UINT16:
		return dtypes.Uint16
	case protos.TensorProto_UINT32:
		return dtypes.Uint32
	case protos.TensorProto_UINT64:
		return dtypes.Uint64
	default:
		return dtypes.InvalidDType
	}
}

// ShapeFromTensorProto converts an ONNX TensorProto into a GoMLX shapes.Shape.
func ShapeFromTensorProto(tp *protos.TensorProto) shapes.Shape {
	if tp == nil {
		return shapes.Invalid()
	}
	dt := DTypeFromONNX(protos.TensorProto_DataType(tp.GetDataType()))
	dims64 := tp.GetDims()
	if dims64 == nil {
		return shapes.Make(dt)
	}
	dims := make([]int, len(dims64))
	for i, d := range dims64 {
		dims[i] = int(d)
	}
	return shapes.Make(dt, dims...)
}

// FormatTensorValues extracts up to maxItems elements from an ONNX TensorProto and formats them as a string.
// If the number of items exceeds maxItems, a "..." suffix is added.
func FormatTensorValues(tp *protos.TensorProto, maxItems int) string {
	if tp == nil {
		return ""
	}
	sh := ShapeFromTensorProto(tp)
	totalSize := sh.Size()
	if totalSize == 0 {
		return "[]"
	}

	var elements []string
	dt := sh.DType

	if len(tp.RawData) > 0 && dt != dtypes.InvalidDType {
		elemSize := dt.Size()
		if elemSize > 0 && len(tp.RawData) >= totalSize*elemSize {
			ptr := unsafe.Pointer(&tp.RawData[0])
			sliceAny := dtypes.UnsafeAnySliceFromBytes(ptr, dt, totalSize)
			elements = sliceToStrings(sliceAny, maxItems)
		}
	}

	if len(elements) == 0 {
		if len(tp.FloatData) > 0 {
			elements = sliceToStrings(tp.FloatData, maxItems)
		} else if len(tp.Int32Data) > 0 {
			elements = sliceToStrings(tp.Int32Data, maxItems)
		} else if len(tp.Int64Data) > 0 {
			elements = sliceToStrings(tp.Int64Data, maxItems)
		} else if len(tp.DoubleData) > 0 {
			elements = sliceToStrings(tp.DoubleData, maxItems)
		} else if len(tp.Uint64Data) > 0 {
			elements = sliceToStrings(tp.Uint64Data, maxItems)
		} else if len(tp.StringData) > 0 {
			elements = sliceToStrings(tp.StringData, maxItems)
		}
	}

	if len(elements) == 0 {
		return "<unparsed data>"
	}

	truncated := totalSize > len(elements)
	if sh.IsScalar() && len(elements) == 1 && !truncated {
		return elements[0]
	}

	res := "[" + strings.Join(elements, ", ")
	if truncated {
		res += ", ..."
	}
	res += "]"
	return res
}

func sliceToStrings(sliceAny any, maxItems int) []string {
	v := reflect.ValueOf(sliceAny)
	if v.Kind() != reflect.Slice && v.Kind() != reflect.Array {
		return nil
	}
	n := v.Len()
	if maxItems >= 0 && n > maxItems {
		n = maxItems
	}
	res := make([]string, n)
	for i := 0; i < n; i++ {
		elem := v.Index(i).Interface()
		switch val := elem.(type) {
		case float16.Float16:
			res[i] = fmt.Sprintf("%g", val.Float32())
		case bfloat16.BFloat16:
			res[i] = fmt.Sprintf("%g", val.Float32())
		case float32:
			res[i] = fmt.Sprintf("%g", val)
		case float64:
			res[i] = fmt.Sprintf("%g", val)
		case []byte:
			res[i] = fmt.Sprintf("%q", string(val))
		case string:
			res[i] = fmt.Sprintf("%q", val)
		default:
			res[i] = fmt.Sprintf("%v", val)
		}
	}
	return res
}

func formatAttribute(attr *protos.AttributeProto, maxItems int) string {
	if attr == nil {
		return ""
	}
	name := attr.GetName()
	switch attr.GetType() {
	case protos.AttributeProto_FLOAT:
		return fmt.Sprintf("%s: %g", name, attr.GetF())
	case protos.AttributeProto_INT:
		return fmt.Sprintf("%s: %d", name, attr.GetI())
	case protos.AttributeProto_STRING:
		return fmt.Sprintf("%s: %q", name, string(attr.GetS()))
	case protos.AttributeProto_TENSOR:
		tp := attr.GetT()
		sh := ShapeFromTensorProto(tp)
		vals := FormatTensorValues(tp, maxItems)
		return fmt.Sprintf("%s: %s = %s", name, sh.String(), vals)
	case protos.AttributeProto_GRAPH:
		g := attr.GetG()
		if g != nil {
			return fmt.Sprintf("%s: Graph(%s)", name, g.GetName())
		}
		return fmt.Sprintf("%s: Graph()", name)
	case protos.AttributeProto_TYPE_PROTO:
		tp := attr.GetTp()
		if sh, ok := ShapeFromONNXType(tp); ok {
			return fmt.Sprintf("%s: %s", name, sh.String())
		}
		return fmt.Sprintf("%s: TypeProto()", name)
	case protos.AttributeProto_FLOATS:
		return fmt.Sprintf("%s: %s", name, formatSlice(attr.GetFloats(), maxItems))
	case protos.AttributeProto_INTS:
		return fmt.Sprintf("%s: %s", name, formatSlice(attr.GetInts(), maxItems))
	case protos.AttributeProto_STRINGS:
		strs := make([]string, len(attr.GetStrings()))
		for i, s := range attr.GetStrings() {
			strs[i] = string(s)
		}
		return fmt.Sprintf("%s: %s", name, formatSlice(strs, maxItems))
	case protos.AttributeProto_TENSORS:
		var tensorStrs []string
		for i, t := range attr.GetTensors() {
			if i >= maxItems && maxItems >= 0 {
				tensorStrs = append(tensorStrs, "...")
				break
			}
			sh := ShapeFromTensorProto(t)
			tensorStrs = append(tensorStrs, fmt.Sprintf("%s = %s", sh.String(), FormatTensorValues(t, maxItems)))
		}
		return fmt.Sprintf("%s: [%s]", name, strings.Join(tensorStrs, ", "))
	default:
		if attr.GetF() != 0 {
			return fmt.Sprintf("%s: %g", name, attr.GetF())
		}
		if attr.GetI() != 0 {
			return fmt.Sprintf("%s: %d", name, attr.GetI())
		}
		if len(attr.GetS()) > 0 {
			return fmt.Sprintf("%s: %q", name, string(attr.GetS()))
		}
		if len(attr.GetInts()) > 0 {
			return fmt.Sprintf("%s: %s", name, formatSlice(attr.GetInts(), maxItems))
		}
		if len(attr.GetFloats()) > 0 {
			return fmt.Sprintf("%s: %s", name, formatSlice(attr.GetFloats(), maxItems))
		}
		if attr.GetT() != nil {
			tp := attr.GetT()
			sh := ShapeFromTensorProto(tp)
			vals := FormatTensorValues(tp, maxItems)
			return fmt.Sprintf("%s: %s = %s", name, sh.String(), vals)
		}
		return fmt.Sprintf("%s", name)
	}
}

func formatSlice[T any](slice []T, maxItems int) string {
	n := len(slice)
	if n == 0 {
		return "[]"
	}
	limit := n
	if maxItems >= 0 && limit > maxItems {
		limit = maxItems
	}
	elems := make([]string, limit)
	for i := 0; i < limit; i++ {
		switch v := any(slice[i]).(type) {
		case float32:
			elems[i] = fmt.Sprintf("%g", v)
		case float64:
			elems[i] = fmt.Sprintf("%g", v)
		case string:
			elems[i] = fmt.Sprintf("%q", v)
		default:
			elems[i] = fmt.Sprintf("%v", v)
		}
	}
	res := "[" + strings.Join(elems, ", ")
	if n > limit {
		res += ", ..."
	}
	res += "]"
	return res
}
