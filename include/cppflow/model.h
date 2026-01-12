// MIT License
//
// Copyright (c) 2020 Sergio Izquierdo
// Copyright (c) 2020 liufeng27
// Copyright (c) 2020 Jiannan Liu
// Copyright (c) 2021 Paolo Galeone
// Copyright (c) 2021 Paul
// Copyright (c) 2022 Tim Upthegrove
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*!
 *  @file       model.h
 *  @author     Jiannan Liu
 *  @author     liufeng27
 *  @author     Paul
 *  @author     Paolo Galeone
 *  @author     Sergio Izquierdo
 *  @author     Tim Upthegrove
 *  @date       @showdate "%B %d, %Y" 2020-06-29
 */

#ifndef INCLUDE_CPPFLOW_MODEL_H_
#define INCLUDE_CPPFLOW_MODEL_H_

// C headers
#include <tensorflow/c/c_api.h>

// C++ headers
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// CppFlow headers
#include "cppflow/context.h"
#include "cppflow/defer.h"
#include "cppflow/tensor.h"
#include "cppflow/pb_helper.h"

namespace cppflow {
  inline void setup_SessionOptions(TF_SessionOptions* options) {

    std::shared_ptr<TF_Status>  status = {TF_NewStatus(), &TF_DeleteStatus};
    // MANUAL PROTOBUF CONSTRUCTION
    // Hierarchy: ConfigProto -> GPUOptions (6) -> Experimental (16) -> DisableTF32 (3)
    // Hex: 32 05 82 01 02 18 00
    const std::vector<uint8_t> config_bytes = {
      0x32, 0x05,             // Field 6 (GPUOptions), Length 5
      0x82, 0x01, 0x02,       // Field 16 (Experimental), Length 2
      0x18, 0x00              // Field 3 (TF32 Enabled), Value 0 (False)
  };

    // Pass the raw bytes to TF_SetConfig
    TF_SetConfig(
        options,
        reinterpret_cast<const void*>(config_bytes.data()),
        config_bytes.size(),
        status.get()
    );

    status_check(status.get());
  }

class model {
 public:
  enum TYPE {
    SAVED_MODEL,
    FROZEN_GRAPH,
  };  // enum TYPE

  explicit model(const std::string& filename,
                 const TYPE type = TYPE::SAVED_MODEL);
  model(const model &model) = default;
  model(model &&model) = default;

  ~model() = default;

  model &operator=(const model &other) = default;
  model &operator=(model &&other) = default;
  std::vector<tensor> operator()(
      std::vector<std::tuple<std::string, tensor>> inputs,
      std::vector<std::string> outputs);
  tensor operator()(const tensor& input);

  std::vector<std::string> get_operations() const;
  std::vector<int64_t> get_operation_shape(const std::string& operation) const;
  void print_signatures();

  std::shared_ptr<TF_Status> status;
  std::shared_ptr<TF_Graph> graph;
  std::shared_ptr<TF_Session> session;

  std::vector<std::string> graph_inputs;
  std::map<string, Signature>  signatures;
  std::vector<std::string> get_graph_inputs() const {return this->graph_inputs;}
  std::vector<std::string> read_graph_inputs() const ;
  std::string get_meta_graph_def() const { return meta_graph_def_; }

    /**
 * Checks if a specific signature key exists in the loaded model.
 * @param name The signature key (e.g., "serving_default")
 * @return true if found, false otherwise.
 */
  bool has_signature(const std::string& name) const {
    // std::map::find returns an iterator to the end if the key is not found
    return signatures.find(name) != signatures.end();
  }

  /**
 * Checks if a specific input tensor name exists in the graph inputs.
 * @param name The input name to check (e.g., "serving_default_input_1")
 * @return true if found, false otherwise.
 */
  bool has_graph_input(const std::string& name) const {
    // Iterate through the vector to find the name
    for (const auto& input : this->graph_inputs) {
      if (input == name) {
        return true;
      }
    }
    return false;
  }

 private:
  TF_Buffer * readGraph(const std::string& filename);
  std::string meta_graph_def_;


};  // Class model

}  // namespace cppflow

namespace cppflow {

  inline model::model(const std::string &filename, const TYPE type) {
    this->status = {TF_NewStatus(), &TF_DeleteStatus};
    this->graph = {TF_NewGraph(), TF_DeleteGraph};

    // Create the session.
    std::unique_ptr<TF_SessionOptions, decltype(&TF_DeleteSessionOptions)>
        session_options = {TF_NewSessionOptions(), TF_DeleteSessionOptions};

    auto session_deleter = [this](TF_Session* sess) {
      TF_DeleteSession(sess, this->status.get());
      status_check(this->status.get());
    };

    setup_SessionOptions(session_options.get());

    if (type == TYPE::SAVED_MODEL) {
      std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> run_options = {
          TF_NewBufferFromString("", 0), TF_DeleteBuffer};
      std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> meta_graph = {
          TF_NewBuffer(), TF_DeleteBuffer};

      int tag_len = 1;
      const char* tag = "serve";
      this->session = {
          TF_LoadSessionFromSavedModel(session_options.get(), run_options.get(),
                                       filename.c_str(), &tag, tag_len,
                                       this->graph.get(), meta_graph.get(),
                                       this->status.get()), session_deleter};
      status_check(this->status.get());

      // --- fill inputs ---
      this->graph_inputs = this->read_graph_inputs();

      // --- fill signatures---
      if (meta_graph->data != nullptr) {
        this->meta_graph_def_.assign(
            static_cast<const char*>(meta_graph->data),
            meta_graph->length
        );

        // 1. Get the raw binary blob
        std::string blob = this->get_meta_graph_def();

        // 2. Parse it
        this->signatures = cppflow::ParseSignatures(blob);
      } else {
        throw std::runtime_error("Failed to import  meta graph data");
      }
      // ----------------------

    } else if (type == TYPE::FROZEN_GRAPH)  {
      this->session = {TF_NewSession(this->graph.get(), session_options.get(),
                                     this->status.get()),
                       session_deleter};
      status_check(this->status.get());

      // Import the graph definition
      TF_Buffer* def = readGraph(filename);
      if (def == nullptr) {
        throw std::runtime_error("Failed to import graph def from file");
      }

      std::unique_ptr<TF_ImportGraphDefOptions,
                      decltype(&TF_DeleteImportGraphDefOptions)> graph_opts = {
          TF_NewImportGraphDefOptions(), TF_DeleteImportGraphDefOptions};
      TF_GraphImportGraphDef(this->graph.get(), def, graph_opts.get(),
                             this->status.get());
      TF_DeleteBuffer(def);
    } else {
      throw std::runtime_error("Model type unknown");
    }

    status_check(this->status.get());
  }

  inline std::vector<std::string> model::get_operations() const {
    std::vector<std::string> result;
    size_t pos = 0;
    TF_Operation* oper;

    // Iterate through the operations of a graph
    while ((oper = TF_GraphNextOperation(this->graph.get(), &pos)) != nullptr) {
      result.emplace_back(TF_OperationName(oper));
    }
    return result;
  }

  inline std::vector<int64_t> model::get_operation_shape(
      const std::string& operation) const {
    // Get operation by the name
    TF_Output out_op;
    out_op.oper = TF_GraphOperationByName(this->graph.get(), operation.c_str());
    out_op.index = 0;

    std::vector<int64_t> shape;

    // Operation does not exist
    if (!out_op.oper)
      throw std::runtime_error("No operation named \"" + operation + "\" exists");

    if (operation == "NoOp")
      throw std::runtime_error("NoOp doesn't have a shape");

    // DIMENSIONS

    // Get number of dimensions
    int n_dims = TF_GraphGetTensorNumDims(this->graph.get(), out_op,
                                          this->status.get());

    // If is not a scalar
    if (n_dims > 0) {
      // Get dimensions
      auto* dims = new int64_t[n_dims];
      TF_GraphGetTensorShape(this->graph.get(), out_op, dims, n_dims,
                             this->status.get());

      // Check error on Model Status
      status_check(this->status.get());

      shape = std::vector<int64_t>(dims, dims + n_dims);

      delete[] dims;
    }

    return shape;
  }

  inline std::tuple<std::string, int> parse_name(const std::string& name) {
    auto idx = name.find(':');
    return (idx == std::string::npos ? std::make_tuple(name, 0) :
            std::make_tuple(name.substr(0, idx),
            std::stoi(name.substr(idx + 1))));
  }

  inline std::vector<tensor> model::operator()(
      std::vector<std::tuple<std::string, tensor>> inputs,
      std::vector<std::string> outputs) {

    std::vector<TF_Output> inp_ops(inputs.size());
    std::vector<TF_Tensor*> inp_val(inputs.size(), nullptr);

    for (decltype(inputs.size()) i=0; i < inputs.size(); i++) {
      // Operations
      const auto[op_name, op_idx] = parse_name(std::get<0>(inputs[i]));
      inp_ops[i].oper = TF_GraphOperationByName(this->graph.get(),
                                                op_name.c_str());
      inp_ops[i].index = op_idx;

      if (!inp_ops[i].oper)
        throw std::runtime_error("No operation named \"" + op_name + "\" exists");

      // Values
      inp_val[i] = std::get<1>(inputs[i]).get_tensor().get();
    }

    std::vector<TF_Output> out_ops(outputs.size());
    auto out_val = std::make_unique<TF_Tensor*[]>(outputs.size());
    for (decltype(outputs.size()) i=0; i < outputs.size(); i++) {
      const auto[op_name, op_idx] = parse_name(outputs[i]);
      out_ops[i].oper = TF_GraphOperationByName(this->graph.get(),
                                                op_name.c_str());
      out_ops[i].index = op_idx;

      if (!out_ops[i].oper)
        throw std::runtime_error("No operation named \"" + op_name + "\" exists");
    }

    TF_SessionRun(this->session.get(), /*run_options*/ NULL,
                  inp_ops.data(), inp_val.data(), static_cast<int>(inputs.size()),
                  out_ops.data(), out_val.get(), static_cast<int>(outputs.size()),
                  /*targets*/ NULL, /*ntargets*/ 0, /*run_metadata*/ NULL,
                  this->status.get());
    status_check(this->status.get());

    std::vector<tensor> result;
    result.reserve(outputs.size());
    for (decltype(outputs.size()) i=0; i < outputs.size(); i++) {
      result.emplace_back(tensor(out_val[i]));
    }

    return result;
  }

  inline tensor model::operator()(const tensor& input) {
    return (*this)({{"serving_default_input_1", input}},
                   {"StatefulPartitionedCall"})[0];
  }

  inline TF_Buffer * model::readGraph(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    // Error opening the file
    if (!file.is_open()) {
      std::cerr << "Unable to open file: " << filename << std::endl;
      return nullptr;
    }

    // Cursor is at the end to get size
    auto size = file.tellg();
    // Move cursor to the beginning
    file.seekg(0, std::ios::beg);

    // Read
    auto data = std::make_unique<char[]>(size);
    file.seekg(0, std::ios::beg);
    file.read(data.get(), size);

    // Error reading the file
    if (!file) {
      std::cerr << "Unable to read the full file: " << filename << std::endl;
      return nullptr;
    }

    // Create tensorflow buffer from read data
    TF_Buffer* buffer = TF_NewBufferFromString(data.get(), size);

    // Close file and remove data
    file.close();

    return buffer;
  }

    // Implementation in namespace cppflow

    inline std::vector<std::string> model::read_graph_inputs() const {
    std::vector<std::string> inputs;
    size_t pos = 0;
    TF_Operation* oper;

    // Iterate through all operations in the graph
    while ((oper = TF_GraphNextOperation(this->graph.get(), &pos)) != nullptr) {
      // Check the operation type
      const char* type = TF_OperationOpType(oper);

      // "Placeholder" nodes are the standard inputs for SavedModels
      if (std::string(type) == "Placeholder") {
        inputs.emplace_back(TF_OperationName(oper));
      }
    }
    return inputs;
  }

  inline void model::print_signatures() {
    for (const auto& [sig_name, sig_data] : signatures) {
      std::cout << "Signature: " << sig_name << std::endl;

      // --- INPUTS ---
      std::cout << "  Inputs:" << std::endl;
      for (const auto& [input_key, tensor_info] : sig_data.inputs) {
        std::cout << "    Key: \"" << input_key << "\"" << std::endl;
        std::cout << "      Tensor: " << tensor_info.name << std::endl;

        // Dtype: Cast int to TF_DataType to use the helper from datatype.h
        std::cout << "      DType:  "
                  << cppflow::to_string(static_cast<TF_DataType>(tensor_info.dtype))
                  << std::endl;

        // Shape: Print vector as [dim, dim, ...]
        std::cout << "      Shape:  [";
        for (size_t i = 0; i < tensor_info.shape.size(); ++i) {
          std::cout << tensor_info.shape[i]
                    << (i < tensor_info.shape.size() - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
      }

      // --- OUTPUTS ---
      std::cout << "  Outputs:" << std::endl;
      for (const auto& [output_key, tensor_info] : sig_data.outputs) {
        std::cout << "    Key: \"" << output_key << "\"" << std::endl;
        std::cout << "      Tensor: " << tensor_info.name << std::endl;

        std::cout << "      DType:  "
                  << cppflow::to_string(static_cast<TF_DataType>(tensor_info.dtype))
                  << std::endl;

        std::cout << "      Shape:  [";
        for (size_t i = 0; i < tensor_info.shape.size(); ++i) {
          std::cout << tensor_info.shape[i]
                    << (i < tensor_info.shape.size() - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
      }
    }
  }
}  // namespace cppflow

#endif  // INCLUDE_CPPFLOW_MODEL_H_
