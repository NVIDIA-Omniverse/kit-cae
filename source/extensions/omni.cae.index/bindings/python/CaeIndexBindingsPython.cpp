// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <carb/BindingsPythonUtils.h>
#include <carb/logging/Log.h>

#include <mi/math/bbox.h>
#include <nv/index/idistributed_compute_destination_buffer.h>
#include <nv/index/idistributed_data_subset.h>
#include <nv/index/iirregular_volume_subset.h>
#include <nv/index/ivdb_subset.h>
#include <omni/cae/data/IFieldArray.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include <cuda_runtime.h>

namespace py = pybind11;

CARB_BINDINGS("omni.cae.data.python")
PYBIND11_DECLARE_HOLDER_TYPE(T, mi::base::Handle<T>, true);

namespace
{

size_t getSizeInBytes(const omni::cae::data::IFieldArray* array)
{
    if (array)
    {
        auto shape = array->getShape();
        return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>{}) *
               omni::cae::data::getElementSize(array->getElementType());
    }
    return 0u;
}


using BboxFloat32 = mi::math::Bbox<mi::Float32, 3>;
using BboxInt32 = mi::math::Bbox<mi::Sint32, 3>;

mi::Uint32 attribute_type_size(const nv::index::IIrregular_volume_subset::Attribute_type& type)
{
    mi::Uint32 sz = 0;
    switch (type)
    {
    case nv::index::IIrregular_volume_subset::ATTRIB_TYPE_UINT8:
        sz = sizeof(mi::Uint8);
        break;
    case nv::index::IIrregular_volume_subset::ATTRIB_TYPE_UINT8_2:
        sz = 2 * sizeof(mi::Uint8);
        break;
    case nv::index::IIrregular_volume_subset::ATTRIB_TYPE_UINT8_3:
        sz = 3 * sizeof(mi::Uint8);
        break;
    case nv::index::IIrregular_volume_subset::ATTRIB_TYPE_UINT8_4:
        sz = 4 * sizeof(mi::Uint8);
        break;
    case nv::index::IIrregular_volume_subset::ATTRIB_TYPE_UINT16:
        sz = sizeof(mi::Uint16);
        break;
    case nv::index::IIrregular_volume_subset::ATTRIB_TYPE_UINT16_2:
        sz = 2 * sizeof(mi::Uint16);
        break;
    case nv::index::IIrregular_volume_subset::ATTRIB_TYPE_UINT16_3:
        sz = 3 * sizeof(mi::Uint16);
        break;
    case nv::index::IIrregular_volume_subset::ATTRIB_TYPE_UINT16_4:
        sz = 4 * sizeof(mi::Uint16);
        break;
    case nv::index::IIrregular_volume_subset::ATTRIB_TYPE_FLOAT32:
        sz = sizeof(mi::Float32);
        break;
    case nv::index::IIrregular_volume_subset::ATTRIB_TYPE_FLOAT32_2:
        sz = 2 * sizeof(mi::Float32);
        break;
    case nv::index::IIrregular_volume_subset::ATTRIB_TYPE_FLOAT32_3:
        sz = 3 * sizeof(mi::Float32);
        break;
    case nv::index::IIrregular_volume_subset::ATTRIB_TYPE_FLOAT32_4:
        sz = 4 * sizeof(mi::Float32);
        break;
    }
    return sz;
}

auto getDType(const nv::index::IIrregular_volume_subset::Attribute_parameters& params)
{
    using Attribute_type = nv::index::IIrregular_volume_subset::Attribute_type;
    switch (params.type)
    {
    case Attribute_type::ATTRIB_TYPE_UINT8:
    case Attribute_type::ATTRIB_TYPE_UINT8_2:
    case Attribute_type::ATTRIB_TYPE_UINT8_3:
    case Attribute_type::ATTRIB_TYPE_UINT8_4:
        return py::dtype::of<uint8_t>();

    case Attribute_type::ATTRIB_TYPE_UINT16:
    case Attribute_type::ATTRIB_TYPE_UINT16_2:
    case Attribute_type::ATTRIB_TYPE_UINT16_3:
    case Attribute_type::ATTRIB_TYPE_UINT16_4:
        return py::dtype::of<uint16_t>();
    case Attribute_type::ATTRIB_TYPE_FLOAT32:
    case Attribute_type::ATTRIB_TYPE_FLOAT32_2:
    case Attribute_type::ATTRIB_TYPE_FLOAT32_3:
    case Attribute_type::ATTRIB_TYPE_FLOAT32_4:
        return py::dtype::of<float>();
    default:
        throw std::runtime_error("Invalid attribute type");
    }
}

pybind11::array::ShapeContainer getShape(const nv::index::IIrregular_volume_subset::Attribute_parameters& params)
{
    using Attribute_type = nv::index::IIrregular_volume_subset::Attribute_type;
    switch (params.type)
    {
    case Attribute_type::ATTRIB_TYPE_UINT8:
    case Attribute_type::ATTRIB_TYPE_UINT16:
    case Attribute_type::ATTRIB_TYPE_FLOAT32:
        return { params.nb_attrib_values };


    case Attribute_type::ATTRIB_TYPE_UINT8_2:
    case Attribute_type::ATTRIB_TYPE_UINT16_2:
    case Attribute_type::ATTRIB_TYPE_FLOAT32_2:
        return { params.nb_attrib_values, 2u };

    case Attribute_type::ATTRIB_TYPE_UINT8_3:
    case Attribute_type::ATTRIB_TYPE_UINT16_3:
    case Attribute_type::ATTRIB_TYPE_FLOAT32_3:
        return { params.nb_attrib_values, 3u };


    case Attribute_type::ATTRIB_TYPE_UINT8_4:
    case Attribute_type::ATTRIB_TYPE_UINT16_4:
    case Attribute_type::ATTRIB_TYPE_FLOAT32_4:
        return { params.nb_attrib_values, 4u };

    default:
        throw std::runtime_error("Invalid attribute type");
    }
}

pybind11::array::StridesContainer getStrides(const nv::index::IIrregular_volume_subset::Attribute_parameters& params)
{
    using Attribute_type = nv::index::IIrregular_volume_subset::Attribute_type;
    switch (params.type)
    {
    case Attribute_type::ATTRIB_TYPE_UINT8:
        return { sizeof(uint8_t) };
    case Attribute_type::ATTRIB_TYPE_UINT16:
        return { sizeof(uint16_t) };
    case Attribute_type::ATTRIB_TYPE_FLOAT32:
        return { sizeof(float) };

    case Attribute_type::ATTRIB_TYPE_UINT8_2:
        return { 2 * sizeof(uint8_t), sizeof(uint8_t) };
    case Attribute_type::ATTRIB_TYPE_UINT16_2:
        return { 2 * sizeof(uint16_t), sizeof(uint16_t) };
    case Attribute_type::ATTRIB_TYPE_FLOAT32_2:
        return { 2 * sizeof(float), sizeof(float) };

    case Attribute_type::ATTRIB_TYPE_UINT8_3:
        return { 3 * sizeof(uint8_t), sizeof(uint8_t) };
    case Attribute_type::ATTRIB_TYPE_UINT16_3:
        return { 3 * sizeof(uint16_t), sizeof(uint16_t) };
    case Attribute_type::ATTRIB_TYPE_FLOAT32_3:
        return { 3 * sizeof(float), sizeof(float) };

    case Attribute_type::ATTRIB_TYPE_UINT8_4:
        return { 4 * sizeof(uint8_t), sizeof(uint8_t) };
    case Attribute_type::ATTRIB_TYPE_UINT16_4:
        return { 4 * sizeof(uint16_t), sizeof(uint16_t) };
    case Attribute_type::ATTRIB_TYPE_FLOAT32_4:
        return { 4 * sizeof(float), sizeof(float) };

    default:
        throw std::runtime_error("Invalid attribute type");
    }
}


PYBIND11_MODULE(_omni_cae_index, m)
{
    m.doc() = "Omni CAE Index Bindings";

    py::class_<BboxFloat32>(m, "Bbox_float32")
        .def(py::init<>())
        .def_property(
            "min", [](const BboxFloat32& bbox) { return py::make_tuple(bbox.min.x, bbox.min.y, bbox.min.z); },
            [](BboxFloat32& bbox, py::iterable iter)
            {
                std::vector<float> vals;
                for (auto v : iter)
                {
                    vals.push_back(v.cast<float>());
                }
                if (vals.size() != 3)
                {
                    throw py::value_error("3-tuple expected");
                }
                std::copy_n(vals.begin(), 3, &bbox.min.x);
            })
        .def_property(
            "max", [](const BboxFloat32& bbox) { return py::make_tuple(bbox.max.x, bbox.max.y, bbox.max.z); },
            [](BboxFloat32& bbox, py::iterable iter)
            {
                std::vector<float> vals;
                for (auto v : iter)
                {
                    vals.push_back(v.cast<float>());
                }
                if (vals.size() != 3)
                {
                    throw py::value_error("3-tuple expected");
                }
                std::copy_n(vals.begin(), 3, &bbox.max.x);
            })
        /**/;

    py::class_<BboxInt32>(m, "Bbox_int32")
        .def(py::init<>())
        .def_property(
            "min", [](const BboxInt32& bbox) { return py::make_tuple(bbox.min.x, bbox.min.y, bbox.min.z); },
            [](BboxInt32& bbox, py::iterable iter)
            {
                std::vector<int> vals;
                for (auto v : iter)
                {
                    vals.push_back(v.cast<int>());
                }
                if (vals.size() != 3)
                {
                    throw py::value_error("3-tuple expected");
                }
                std::copy_n(vals.begin(), 3, &bbox.min.x);
            })
        .def_property(
            "max", [](const BboxInt32& bbox) { return py::make_tuple(bbox.max.x, bbox.max.y, bbox.max.z); },
            [](BboxInt32& bbox, py::iterable iter)
            {
                std::vector<int> vals;
                for (auto v : iter)
                {
                    vals.push_back(v.cast<int>());
                }
                if (vals.size() != 3)
                {
                    throw py::value_error("3-tuple expected");
                }
                std::copy_n(vals.begin(), 3, &bbox.max.x);
            })
        /**/;

    using Mesh_parameters = nv::index::IIrregular_volume_subset::Mesh_parameters;
    py::class_<Mesh_parameters>(m, "Mesh_parameters")
        .def(py::init<>())
        .def_readwrite("nb_vertices", &Mesh_parameters::nb_vertices, "Size of the vertex array in number of vertices.")
        .def_readwrite("nb_face_vtx_indices", &Mesh_parameters::nb_face_vtx_indices,
                       "Size of the face vertex index array in number of elements.")
        .def_readwrite("nb_faces", &Mesh_parameters::nb_faces, "Size of the face array in number of faces.")
        .def_readwrite("nb_cell_face_indices", &Mesh_parameters::nb_cell_face_indices,
                       "Size of the cell face index array in number of elements.")
        .def_readwrite("nb_cells", &Mesh_parameters::nb_cells, "Size of the cell array in number of faces.")
        .def_readwrite("global_max_edge_length", &Mesh_parameters::global_max_edge_length,
                       "The length of the longest edge in the irregular volume mesh ")
        .def("__str__",
             [](const Mesh_parameters& self)
             {
                 std::ostringstream str;
                 str << "Mesh_parameters(" << &self << "):\n";
                 str << "  nb_vertices=" << self.nb_vertices << "\n";
                 str << "  nb_face_vtx_indices=" << self.nb_face_vtx_indices << "\n";
                 str << "  nb_faces=" << self.nb_faces << "\n";
                 str << "  nb_cell_face_indices=" << self.nb_cell_face_indices << "\n";
                 str << "  nb_cells=" << self.nb_cells << "\n";
                 str << "  global_max_edge_length=" << self.global_max_edge_length << "\n";
                 return str.str();
             })
        /**/;

    using Mesh_storage = nv::index::IIrregular_volume_subset::Mesh_storage;
    py::class_<Mesh_storage>(m, "Mesh_storage")
        .def(
            "get_vertices",
            [](Mesh_storage& self, const Mesh_parameters& params)
            {
                py::capsule capsule([]() {});
                std::array<py::ssize_t, 2> shape = { params.nb_vertices, 3u };
                std::array<py::ssize_t, 2> strides = { 3 * sizeof(mi::Float32), sizeof(mi::Float32) };
                return py::array(
                    py::dtype::of<mi::Float32>(), shape, strides, static_cast<void*>(&self.vertices->x), capsule);
            },
            "The vertex array.")
        .def(
            "get_face_vtx_indices",
            [](Mesh_storage& self, const Mesh_parameters& params)
            {
                py::capsule capsule([]() {});
                return py::array(py::dtype::of<mi::Uint32>(), { params.nb_face_vtx_indices }, { sizeof(mi::Uint32) },
                                 self.face_vtx_indices, capsule);
            },
            "The face-vertex index array.")
        .def(
            "get_faces",
            [](Mesh_storage& self, const Mesh_parameters& params)
            {
                py::capsule capsule([]() {});
                std::array<py::ssize_t, 2> shape = { params.nb_faces, 2u };
                std::array<py::ssize_t, 2> strides = { 2 * sizeof(mi::Uint32), sizeof(mi::Uint32) };
                return py::array(py::dtype::of<mi::Uint32>(), shape, strides, static_cast<void*>(self.faces), capsule);
            },
            "The face array.")
        .def(
            "get_cell_face_indices",
            [](Mesh_storage& self, const Mesh_parameters& params)
            {
                py::capsule capsule([]() {});
                return py::array(py::dtype::of<mi::Uint32>(), { params.nb_cell_face_indices }, { sizeof(mi::Uint32) },
                                 self.cell_face_indices, capsule);
            },
            "The cell face indices array.")
        .def(
            "get_cells",
            [](Mesh_storage& self, const Mesh_parameters& params)
            {
                py::capsule capsule([]() {});
                std::array<py::ssize_t, 2> shape = { params.nb_cells, 2u };
                std::array<py::ssize_t, 2> strides = { 2 * sizeof(mi::Uint32), sizeof(mi::Uint32) };
                return py::array(py::dtype::of<mi::Uint32>(), shape, strides, static_cast<void*>(self.cells), capsule);
            },
            "The cell array.")
        /**/;

    using Attribute_parameters = nv::index::IIrregular_volume_subset::Attribute_parameters;
    py::class_<Attribute_parameters>(m, "Attribute_parameters")
        .def(py::init<>())
        .def_readwrite("affiliation", &Attribute_parameters::affiliation, "Attribute affiliation.")
        .def_readwrite("type", &Attribute_parameters::type, "Attribute type.")
        .def_readwrite("nb_attrib_values", &Attribute_parameters::nb_attrib_values,
                       "Number of attribute values in number of elements.")
        /**/;

    using Attribute_affiliation = nv::index::IIrregular_volume_subset::Attribute_affiliation;
    py::enum_<Attribute_affiliation>(m, "Attribute_affiliation")
        .value("ATTRIB_AFFIL_PER_VERTEX", Attribute_affiliation::ATTRIB_AFFIL_PER_VERTEX)
        .value("ATTRIB_AFFIL_PER_CELL", Attribute_affiliation::ATTRIB_AFFIL_PER_CELL)
        .export_values();

    using Attribute_type = nv::index::IIrregular_volume_subset::Attribute_type;
    py::enum_<Attribute_type>(m, "Attribute_type")
        .value("ATTRIB_TYPE_UINT8", Attribute_type::ATTRIB_TYPE_UINT8)
        .value("ATTRIB_TYPE_UINT8_2", Attribute_type::ATTRIB_TYPE_UINT8_2)
        .value("ATTRIB_TYPE_UINT8_3", Attribute_type::ATTRIB_TYPE_UINT8_3)
        .value("ATTRIB_TYPE_UINT8_4", Attribute_type::ATTRIB_TYPE_UINT8_4)
        .value("ATTRIB_TYPE_UINT16", Attribute_type::ATTRIB_TYPE_UINT16)
        .value("ATTRIB_TYPE_UINT16_2", Attribute_type::ATTRIB_TYPE_UINT16_2)
        .value("ATTRIB_TYPE_UINT16_3", Attribute_type::ATTRIB_TYPE_UINT16_3)
        .value("ATTRIB_TYPE_UINT16_4", Attribute_type::ATTRIB_TYPE_UINT16_4)
        .value("ATTRIB_TYPE_FLOAT32", Attribute_type::ATTRIB_TYPE_FLOAT32)
        .value("ATTRIB_TYPE_FLOAT32_2", Attribute_type::ATTRIB_TYPE_FLOAT32_2)
        .value("ATTRIB_TYPE_FLOAT32_3", Attribute_type::ATTRIB_TYPE_FLOAT32_3)
        .value("ATTRIB_TYPE_FLOAT32_4", Attribute_type::ATTRIB_TYPE_FLOAT32_4)
        .export_values();

    using Attribute_storage = nv::index::IIrregular_volume_subset::Attribute_storage;
    py::class_<Attribute_storage>(m, "Attribute_storage")
        .def(py::init<>())
        .def(
            "get_attrib_values",
            [](Attribute_storage& self, const Attribute_parameters& params)
            {
                py::capsule capsule([]() {});
                return py::array(getDType(params), getShape(params), getStrides(params), self.attrib_values, capsule);
            },
            "The attribute array.")
        /**/;

    // Declare base class first - REQUIRED for inheritance to work in pybind11
    // Using mi::base::Handle as holder type - prevents pybind11 from calling delete on MI interfaces
    using nv::index::IDistributed_data_subset;
    py::class_<IDistributed_data_subset, mi::base::Handle<IDistributed_data_subset>>(m, "IDistributed_data_subset")
        /**/;

    // Declare derived class with base class - this tells pybind11 about the inheritance
    using IIrregular_volume_subset = nv::index::IIrregular_volume_subset;
    py::class_<IIrregular_volume_subset, IDistributed_data_subset, mi::base::Handle<IIrregular_volume_subset>>(
        m, "IIrregular_volume_subset")
        .def(
            "generate_mesh_storage",
            [](IIrregular_volume_subset* self, const Mesh_parameters& params)
            {
                py::gil_scoped_release g;
                Mesh_storage storage;
                if (!self->generate_mesh_storage(params, storage))
                {
                    throw std::runtime_error("Failed to generate mesh storage");
                }
                return storage;
            },
            "Generate and initialize an instance of irregular volume mesh storage.")
        .def(
            "generate_attribute_storage",
            [](IIrregular_volume_subset* self, mi::Uint32 index, const Attribute_parameters& params)
            {
                py::gil_scoped_release g;
                Attribute_storage storage;
                if (!self->generate_attribute_storage(index, params, storage))
                {
                    throw std::runtime_error("Failed to generate attribute storage.");
                }
                return storage;
            },
            "Generate and initialize an instance of irregular volume attribute set storage.")
        .def(
            "get_attribute_parameters",
            [](IIrregular_volume_subset* self, mi::Uint32 index) -> Attribute_parameters
            {
                py::gil_scoped_release g;
                Attribute_parameters params;
                if (!self->get_attribute_parameters(index, params))
                {
                    throw std::runtime_error("Failed to get attribute parameters.");
                }
                return params;
            },
            "Get the attribute parameters for the given index.")

        .def(
            "get_attribute",
            [](IIrregular_volume_subset* self, mi::Uint32 index) -> Attribute_storage
            {
                py::gil_scoped_release g;
                Attribute_storage storage;
                if (!self->get_attribute(index, storage))
                {
                    throw std::runtime_error("Failed to get attribute storage.");
                }
                return storage;
            },
            "Get the attribute storage for the given index.")
        .def(
            "sync_device_storage",
            [](IIrregular_volume_subset* self, mi::Uint32 index)
            {
                py::gil_scoped_release g;
                Attribute_storage host_storage;
                if (!self->get_attribute(index, host_storage))
                {
                    throw std::runtime_error("Failed to get attribute storage.");
                }
                Attribute_storage device_storage;
                if (!self->get_active_attribute_device_storage(index, device_storage))
                {
                    throw std::runtime_error("Failed to get active attribute device storage.");
                }

                Attribute_parameters params;
                if (!self->get_attribute_parameters(index, params))
                {
                    throw std::runtime_error("Failed to get attribute parameters.");
                }

                auto nb_bytes = mi::Size(params.nb_attrib_values) * attribute_type_size(params.type);
                cudaMemcpy(device_storage.attrib_values, host_storage.attrib_values, nb_bytes, cudaMemcpyHostToDevice);
            },
            "Sync the device storage for the given index.")
        /**/;

    using IVDB_subset = nv::index::IVDB_subset;
    using IVDB_subset_device = nv::index::IVDB_subset_device;
    py::class_<IVDB_subset, IDistributed_data_subset, mi::base::Handle<IVDB_subset>>(m, "IVDB_subset")
        .def(
            "get_device_subset",
            [](IVDB_subset* self) -> mi::base::Handle<IVDB_subset_device>
            {
                py::gil_scoped_release g;
                auto handle = mi::base::make_handle(self->get_device_subset());
                if (!handle)
                {
                    throw std::runtime_error("Failed to get device subset.");
                }
                return handle;
            },
            "Get the device subset.")
        /**/;

    using omni::cae::data::IFieldArray;
    py::class_<IVDB_subset_device, mi::base::Handle<IVDB_subset_device>>(m, "IVDB_subset_device")
        .def(
            "get_device_id",
            [](IVDB_subset_device* self) -> mi::Uint32
            {
                py::gil_scoped_release g;
                return self->get_device_id();
            },
            "Get the device id.")
        .def(
            "adopt_field_array",
            [](IVDB_subset_device* self, mi::Uint32 index, carb::ObjectPtr<IFieldArray> field_array)
            {
                py::gil_scoped_release g;

                /// TODO: copy to correct device, if needed
                if (!self->adopt_grid_buffer(
                        index, const_cast<void*>(field_array->getData()), getSizeInBytes(field_array.get())))
                {
                    throw std::runtime_error("Failed to adopt field array.");
                }
            },
            "Adopt the field array.")
        /**/;
    using nv::index::IData_subset_factory;
    py::class_<IData_subset_factory>(m, "IData_subset_factory")
        .def(
            "create_irregular_volume_subset",
            [](IData_subset_factory* self) -> mi::base::Handle<IIrregular_volume_subset>
            {
                py::gil_scoped_release g;
                auto subset = mi::base::make_handle(self->create_data_subset<nv::index::IIrregular_volume_subset>());
                if (!subset)
                {
                    throw std::runtime_error("Failed to create irregular volume subset.");
                }
                return subset;
            },
            "Create an empty irregular volume subset.")
        /**/;

    using nv::index::IDistributed_compute_destination_buffer;
    using nv::index::IDistributed_compute_destination_buffer_irregular_volume;
    using nv::index::IDistributed_compute_destination_buffer_VDB;


    py::class_<IDistributed_compute_destination_buffer>(m, "IDistributed_compute_destination_buffer")
        /**/;

    py::class_<IDistributed_compute_destination_buffer_irregular_volume, IDistributed_compute_destination_buffer>(
        m, "IDistributed_compute_destination_buffer_irregular_volume")
        .def(
            "get_distributed_data_subset",
            [](IDistributed_compute_destination_buffer_irregular_volume* self) -> mi::base::Handle<IIrregular_volume_subset>
            {
                py::gil_scoped_release g;
                auto handle = mi::base::make_handle(self->get_distributed_data_subset());
                if (!handle)
                {
                    throw std::runtime_error("Failed to get distributed data subset for irregular volume.");
                }
                return handle;
            },
            "Get the distributed data subset ().")
        /**/;

    py::class_<IDistributed_compute_destination_buffer_VDB, IDistributed_compute_destination_buffer>(
        m, "IDistributed_compute_destination_buffer_VDB")
        .def(
            "get_distributed_data_subset",
            [](IDistributed_compute_destination_buffer_VDB* self) -> mi::base::Handle<IVDB_subset>
            {
                py::gil_scoped_release g;
                auto handle = mi::base::make_handle(self->get_distributed_data_subset());
                if (!handle)
                {
                    throw std::runtime_error("Failed to get distributed data subset for vdb.");
                }
                return handle;
            },
            "Get the distributed data subset.")
        /**/;
}

} // namespace {}
