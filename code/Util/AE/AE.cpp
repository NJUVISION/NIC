/**************************************************
	> File Name:  AE.cpp
	> Author:     Leuckart
	> Time:       2019-09-29 01:33
**************************************************/

#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>

// replace this with your specific <Python.h> path
#include </opt/conda/include/python3.6m/Python.h>

#include "My_Range_Coder.h"

#define PRECISION 16

static RangeEncoder arithmetic_encoder;
static RangeDecoder arithmetic_decoder;

bool My_Encode(const std::vector<int32_t> &Datas,
			   const std::vector<double_t> &Means,
			   const std::vector<double_t> &Scales,
			   const char *file_name)
{
	arithmetic_encoder.Init(file_name);

	const int32_t data_min = *min_element(Datas.cbegin(), Datas.cend());
	const int32_t data_max = *max_element(Datas.cbegin(), Datas.cend());
	std::cout << " C++ ===>  Min: " << data_min << " ;  Max: " << data_max << std::endl;

	const uint8_t range_size = 1 + data_max - data_min;
	const uint16_t factor = (1 << PRECISION) - range_size;

	for (size_t i = 0; i < Datas.size(); ++i)
	{
		const uint8_t cur_data = Datas[i] - data_min;
		const double_t cur_mean = Means[i] - data_min, cur_scale = Scales[i];

		const uint32_t x_0 = normalCDF((cur_data - 0.5 - cur_mean) / cur_scale) * factor + cur_data;
		const uint32_t x_1 = normalCDF((cur_data + 0.5 - cur_mean) / cur_scale) * factor + cur_data + 1;
		arithmetic_encoder.Encode(x_0, x_1);
	}
	arithmetic_encoder.Finalize();
	return 0;
}

static PyObject *AE_Encode(PyObject *self, PyObject *args)
{
	char *file_name;
	PyObject *obj_datas, *obj_means, *obj_scales;
	std::vector<int32_t> Datas;
	std::vector<double_t> Means, Scales;

	if (!PyArg_ParseTuple(args, "OOOs", &obj_datas, &obj_means, &obj_scales, &file_name))
		return nullptr;

	// ================================================== //
	PyObject *iter_datas = PyObject_GetIter(obj_datas);
	if (!iter_datas)
	{
		PyErr_SetString(PyExc_TypeError, "The Data Object Is Not Iterable! By Leuckart.\n");
		return NULL;
	}
	while (PyObject *next = PyIter_Next(iter_datas))
	{
		if (!PyLong_Check(next))
		{
			PyErr_SetString(PyExc_TypeError, "Int Or Long Data List is Expected! By Leuckart.\n");
			return NULL;
		}
		Datas.push_back(static_cast<int32_t>(PyLong_AsLong(next)));
	}

	// ================================================== //
	PyObject *iter_means = PyObject_GetIter(obj_means);
	if (!iter_means)
	{
		PyErr_SetString(PyExc_TypeError, "The Mean Object Is Not Iterable! By Leuckart.\n");
		return NULL;
	}
	while (PyObject *next = PyIter_Next(iter_means))
	{
		if (!PyFloat_Check(next))
		{
			PyErr_SetString(PyExc_TypeError, "Float Or Double Mean List is Expected! By Leuckart.\n");
			return NULL;
		}
		Means.push_back(static_cast<double_t>(PyFloat_AS_DOUBLE(next)));
	}

	// ================================================== //
	PyObject *iter_scales = PyObject_GetIter(obj_scales);
	if (!iter_scales)
	{
		PyErr_SetString(PyExc_TypeError, "The Scale Object Is Not Iterable! By Leuckart.\n");
		return NULL;
	}
	while (PyObject *next = PyIter_Next(iter_scales))
	{
		if (!PyFloat_Check(next))
		{
			PyErr_SetString(PyExc_TypeError, "Float Or Double Scale List is Expected! By Leuckart.\n");
			return NULL;
		}
		Scales.push_back(static_cast<double_t>(PyFloat_AS_DOUBLE(next)));
	}

	// ================================================== //
	return (PyObject *)Py_BuildValue("b", My_Encode(Datas, Means, Scales, file_name));
}

int32_t My_Decode(const double_t mean, const double_t scale)
{
	int32_t decoded = arithmetic_decoder.Decode(mean, scale);
	return decoded;
}

static PyObject *AE_Decode(PyObject *self, PyObject *args)
{
	double_t mean, scale;
	if (!PyArg_ParseTuple(args, "dd", &mean, &scale))
		return NULL;

	return (PyObject *)Py_BuildValue("i", My_Decode(mean, scale));
}

bool My_Encode_Cdf(const std::vector<uint32_t> &cdf_0, const std::vector<uint32_t> &cdf_1, const char *file_name)
{
	arithmetic_encoder.Init(file_name);
	for (size_t i = 0; i < cdf_0.size(); ++i)
		arithmetic_encoder.Encode(cdf_0[i], cdf_1[i]);

	arithmetic_encoder.Finalize();
	return 0;
}

static PyObject *AE_Encode_Cdf(PyObject *self, PyObject *args)
{
	char *file_name;
	PyObject *obj_cdf_0, *obj_cdf_1;
	std::vector<uint32_t> cdf_0, cdf_1;

	if (!PyArg_ParseTuple(args, "OOs", &obj_cdf_0, &obj_cdf_1, &file_name))
		return NULL;

	// ================================================== //
	PyObject *iter_cdf_0 = PyObject_GetIter(obj_cdf_0);
	if (!iter_cdf_0)
	{
		PyErr_SetString(PyExc_TypeError, "The Cdf_0 Object Is Not Iterable! By Leuckart.\n");
		return NULL;
	}
	while (PyObject *next = PyIter_Next(iter_cdf_0))
	{
		if (!PyLong_Check(next))
		{
			PyErr_SetString(PyExc_TypeError, "Int Or Long Cdf_0 List is Expected! By Leuckart.\n");
			return NULL;
		}
		cdf_0.push_back(static_cast<uint32_t>(PyLong_AsLong(next)));
	}

	// ================================================== //
	PyObject *iter_cdf_1 = PyObject_GetIter(obj_cdf_1);
	if (!iter_cdf_1)
	{
		PyErr_SetString(PyExc_TypeError, "The Cdf_1 Object Is Not Iterable! By Leuckart.\n");
		return NULL;
	}
	while (PyObject *next = PyIter_Next(iter_cdf_1))
	{
		if (!PyLong_Check(next))
		{
			PyErr_SetString(PyExc_TypeError, "Int Or Long Cdf_1 List is Expected! By Leuckart.\n");
			return NULL;
		}
		cdf_1.push_back(static_cast<uint32_t>(PyLong_AsLong(next)));
	}

	// ================================================== //
	return (PyObject *)Py_BuildValue("b", My_Encode_Cdf(cdf_0, cdf_1, file_name));
}

int32_t My_Decode_Cdf(const std::vector<uint32_t> &cdf)
{
	int32_t decoded = arithmetic_decoder.Decode_Cdf(cdf);
	return decoded;
}

static PyObject *AE_Decode_Cdf(PyObject *self, PyObject *args)
{
	PyObject *obj_cdf;
	std::vector<uint32_t> cdf;

	if (!PyArg_ParseTuple(args, "O", &obj_cdf))
		return NULL;

	// ================================================== //
	PyObject *iter_cdf = PyObject_GetIter(obj_cdf);
	if (!iter_cdf)
	{
		PyErr_SetString(PyExc_TypeError, "The Cdf Object Is Not Iterable! By Leuckart.\n");
		return NULL;
	}
	while (PyObject *next = PyIter_Next(iter_cdf))
	{
		if (!PyLong_Check(next))
		{
			PyErr_SetString(PyExc_TypeError, "Int Or Long Cdf List is Expected! By Leuckart.\n");
			return NULL;
		}
		cdf.push_back(static_cast<uint32_t>(PyLong_AsLong(next)));
	}

	// ================================================== //
	return (PyObject *)Py_BuildValue("i", My_Decode_Cdf(cdf));
}

bool My_Init_Decoder(const char *file_name, const int32_t data_min, const int32_t data_max)
{
	arithmetic_decoder.Init(file_name, data_min, data_max);
	return 0;
}

static PyObject *AE_Init_Decoder(PyObject *self, PyObject *args)
{
	int32_t data_min, data_max;
	char *file_name;
	if (!PyArg_ParseTuple(args, "sii", &file_name, &data_min, &data_max))
	{
		return NULL;
	}

	return (PyObject *)Py_BuildValue("b", My_Init_Decoder(file_name, data_min, data_max));
}

static PyMethodDef AEMethods[] = {
	{"init_decoder", AE_Init_Decoder, METH_VARARGS},
	{"encode", AE_Encode, METH_VARARGS},
	{"encode_cdf", AE_Encode_Cdf, METH_VARARGS},
	{"decode", AE_Decode, METH_VARARGS},
	{"decode_cdf", AE_Decode_Cdf, METH_VARARGS},
	{NULL, NULL},
};

static struct PyModuleDef AEModule = {
	PyModuleDef_HEAD_INIT,
	"AE",
	NULL,
	-1,
	AEMethods};

extern "C"
{
	void PyInit_AE()
	{
		PyModule_Create(&AEModule);
	}
}
