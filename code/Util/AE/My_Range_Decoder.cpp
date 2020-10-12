/**************************************************
	> File Name:  My_Range_Decoder.cpp
	> Author:     Leuckart
	> Time:       2019-11-27 10:46
**************************************************/

#include "My_Range_Coder.h"

RangeDecoder::RangeDecoder(const uint8_t precision) : _precision(precision)
{
	assert(_precision > 0);
	assert(_precision < 17);
}

void RangeDecoder::Init(const std::string filename, const int32_t data_min, const int32_t data_max)
{
	_data_min = data_min;
	_data_max = data_max;
	_factor = (1 << _precision) - (data_max - data_min + 1);

	_base = 0;
	_size_minus1 = std::numeric_limits<uint32_t>::max();
	_value = 0;
	_bit_stream.clear();

	std::ifstream in_file(filename, std::ios::in | std::ios::binary);
	char _rchar;
	while (in_file.read(&_rchar, sizeof(char)))
		_bit_stream.push_back(_rchar);
	in_file.close();

	_current = _bit_stream.cbegin();
	_end = _bit_stream.cend();

	Read16BitValue();
	Read16BitValue();
}

int32_t RangeDecoder::Decode_Cdf(const std::vector<uint32_t> cdf)
{
	assert(cdf.size() > 1);
	const uint64_t size = static_cast<uint64_t>(_size_minus1) + 1;
	const uint64_t offset = ((static_cast<uint64_t>(_value - _base) + 1) << _precision) - 1;

	const uint32_t *ptr_low = cdf.data() + 1;
	std::vector<int32_t>::size_type len = cdf.size() - 1;

	do
	{
		const std::vector<int32_t>::size_type half = len / 2;
		const uint32_t *ptr_mid = ptr_low + half;
		assert(*ptr_mid >= 0);
		//assert(*ptr_mid <= (1 << _precision));

		if (size * static_cast<uint64_t>(*ptr_mid) <= offset)
		{
			ptr_low = ptr_mid + 1;
			len -= half + 1;
		}
		else
		{
			len = half;
		}
	} while (len > 0);

	assert(ptr_low < cdf.data() + cdf.size());

	const uint32_t a = (size * static_cast<uint64_t>(*(ptr_low - 1))) >> _precision;
	const uint32_t b = ((size * static_cast<uint64_t>(*ptr_low)) >> _precision) - 1;
	assert(a <= (offset >> _precision));
	assert((offset >> _precision) <= b);

	_base += a;
	_size_minus1 = b - a;

	if (_size_minus1 >> 16 == 0)
	{
		_base <<= 16;
		_size_minus1 <<= 16;
		_size_minus1 |= 0xFFFF;

		Read16BitValue();
	}

	return ptr_low - cdf.data() - 1 + _data_min;
}

int32_t RangeDecoder::Decode(double_t mean, double_t scale)
{
	mean -= _data_min;
	const uint64_t size = static_cast<uint64_t>(_size_minus1) + 1;
	const uint64_t offset = ((static_cast<uint64_t>(_value - _base) + 1) << _precision) - 1;

	uint32_t low = 1;
	int8_t len = (1 << _precision) - _factor; // data_max - data_min + 1

	do
	{
		const int8_t bias = len / 2;
		uint32_t mid = low + bias;
		const uint32_t mid_val = mid + normalCDF((mid - 0.5 - mean) / scale) * _factor;

		if (size * mid_val <= offset)
		{
			low = mid + 1;
			len -= bias + 1;
		}
		else
		{
			len = bias;
		}
	} while (len > 0);
	low -= 1;

	const uint32_t _a = low + normalCDF((low - 0.5 - mean) / scale) * _factor;
	const uint32_t a = (size * _a) >> _precision;
	const uint32_t _b = low + 1 + normalCDF((low + 0.5 - mean) / scale) * _factor;
	const uint32_t b = ((size * _b) >> _precision) - 1;
	assert(a <= (offset >> _precision));
	assert((offset >> _precision) <= b);

	_base += a;
	_size_minus1 = b - a;

	if (_size_minus1 >> 16 == 0)
	{
		_base <<= 16;
		_size_minus1 <<= 16;
		_size_minus1 |= 0xFFFF;

		Read16BitValue();
	}
	return low + _data_min; 
}

void RangeDecoder::Read16BitValue()
{
	_value <<= 8;
	if (_current != _end)
	{
		_value |= static_cast<uint8_t>(*_current++);
	}
	_value <<= 8;
	if (_current != _end)
	{
		_value |= static_cast<uint8_t>(*_current++);
	}
}
