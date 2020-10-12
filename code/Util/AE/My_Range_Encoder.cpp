/**************************************************
	> File Name:  My_Range_Encoder.cpp
	> Author:     Leuckart
	> Time:       2019-09-27 11:35
**************************************************/

#include "My_Range_Coder.h"

RangeEncoder::RangeEncoder(const uint8_t precision) : _precision(precision)
{
	assert(_precision > 0);
	assert(_precision < 17);
}

void RangeEncoder::Init(const std::string filename)
{
	_out_writer.open(filename, std::ios::out | std::ios::binary);
	std::cout << " C++ ===>  " << filename << std::endl;
	assert(_out_writer);

	_base = 0;
	_size_minus1 = std::numeric_limits<uint32_t>::max();
	_delay = 0;
	_bit_stream.clear();
}

std::vector<char> RangeEncoder::Get_Stream() const
{
	return _bit_stream;
}

void RangeEncoder::Encode_Cdf(const uint8_t x, const std::vector<uint32_t> cdf)
{
	const uint32_t lower = cdf[x], upper = cdf[x + 1];
	this->Encode(lower, upper);
}

void RangeEncoder::Encode(const uint32_t lower, const uint32_t upper)
{
	assert(lower >= 0);
	assert(lower < upper);
	//assert(upper <= (1 << _precision));

	const uint64_t size = static_cast<uint64_t>(_size_minus1) + 1;
	assert(size >> 16);

	const uint32_t a = (size * lower) >> _precision;
	const uint32_t b = ((size * upper) >> _precision) - 1;
	assert(a <= b);

	_base += a;
	_size_minus1 = b - a;
	const bool base_overflow = (_base < a);

	if (_base + _size_minus1 < _base)
	{
		assert(((_base - a) + size) >> 32);
		assert(_delay & 0xFFFF);

		if (_size_minus1 >> 16 == 0)
		{
			assert((_base >> 16) == 0xFFFF);
			_base <<= 16;
			_size_minus1 <<= 16;
			_size_minus1 |= 0xFFFF;

			assert(_delay < (1ULL << 62));
			_delay += 0x20000; // Two more bytes of zeros. Check overflow?
		}
		return;
	}

	if (_delay != 0)
	{
		if (base_overflow)
		{
			assert((static_cast<uint64_t>(_base - a) + a) >> 32);
			_bit_stream.push_back(static_cast<char>(_delay >> 8));
			_bit_stream.push_back(static_cast<char>(_delay >> 0));
			//_bit_stream.append(_delay >> 16, static_cast<char>(0));
			// TODO: 可以不用循环，使用insert合并两个vector
			for (size_t i = 0; i < (_delay >> 16); ++i)
				_bit_stream.push_back(static_cast<char>(0));
		}
		else
		{
			assert((static_cast<uint64_t>(_base + _size_minus1) >> 32) == 0);
			--_delay;
			_bit_stream.push_back(static_cast<char>(_delay >> 8));
			_bit_stream.push_back(static_cast<char>(_delay >> 0));
			//_bit_stream.append(_delay >> 16, static_cast<char>(0xFF));
			for (size_t i = 0; i < (_delay >> 16); ++i)
				_bit_stream.push_back(static_cast<char>(0xFF));
		}
		_delay = 0;
	}

	if ((_size_minus1 >> 16) == 0)
	{
		const uint32_t top = _base >> 16;

		_base <<= 16;
		_size_minus1 <<= 16;
		_size_minus1 |= 0xFFFF;

		if (_base <= _base + _size_minus1)
		{
			_bit_stream.push_back(static_cast<char>(top >> 8));
			_bit_stream.push_back(static_cast<char>(top));
		}
		else
		{
			assert(top < 0xFFFF);
			_delay = top + 1;
		}
	}
}

void RangeEncoder::Finalize()
{
	if (_delay)
	{
		_bit_stream.push_back(static_cast<char>(_delay >> 8));
		if (_delay & 0xFF)
		{
			_bit_stream.push_back(static_cast<char>(_delay));
		}
	}
	else if (_base)
	{
		const uint32_t mid = ((_base - 1) >> 16) + 1;
		assert((mid & 0xFFFF) == mid);
		_bit_stream.push_back(static_cast<char>(mid >> 8));
		if (mid & 0xFF)
		{
			_bit_stream.push_back(static_cast<char>(mid >> 0));
		}
	}

	// TODO: Const? Reference?
	//*
	for (const char item : _bit_stream)
		//_out_writer.write(reinterpret_cast<char *>(&item), sizeof(unsigned char));
		_out_writer.write(&item, sizeof(unsigned char));
	_out_writer.close();
	//*/
	//std::cout << " ===>  " << filename << "  :  " << _bit_stream.size() << " Byte" << std::endl;
	std::cout << " C++ ===>  " << _bit_stream.size() << " Byte" << std::endl;
}
