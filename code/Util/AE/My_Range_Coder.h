/**************************************************
	> File Name:  My_Range_Coder.h
	> Author:     Leuckart
	> Time:       2019-09-27 11:35
**************************************************/

#ifndef MY_RANGE_CODER_H_
#define MY_RANGE_CODER_H_

//#define NDEBUG
#include <limits>
#include <vector>
#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <cmath>

inline double normalCDF(double_t value)
{
    // M_SQRT1_2 ---- 2 的平方根的倒数
    return 0.5 * erfc(-value * M_SQRT1_2);
}

class RangeEncoder
{
public:
    RangeEncoder(const uint8_t precision = 16);
    void Init(const std::string filename);

    void Encode(const uint32_t lower, const uint32_t upper);
    void Encode_Cdf(const uint8_t x, const std::vector<uint32_t> cdf);
    std::vector<char> Get_Stream() const;
    void Finalize();

private:
    // TODO: 可以保存一个文件指针，这样就不用到最后才一个个写入了。
    const uint8_t _precision;

    uint32_t _base = 0;
    uint32_t _size_minus1 = std::numeric_limits<uint32_t>::max();
    uint64_t _delay = 0;
    std::ofstream _out_writer;
    std::vector<char> _bit_stream;
};

class RangeDecoder
{
public:
    RangeDecoder(const uint8_t precision = 16);
    void Init(const std::string filename, const int32_t data_min, const int32_t data_max);
    int32_t Decode(double_t mean, double_t scale);
    int32_t Decode_Cdf(const std::vector<uint32_t> cdf);

private:
    void Read16BitValue();

    const uint8_t _precision;
    int32_t _data_min;
    int32_t _data_max;
    uint16_t _factor;

    uint32_t _base = 0;
    uint32_t _size_minus1 = std::numeric_limits<uint32_t>::max();
    uint32_t _value = 0;

    std::vector<char> _bit_stream;
    std::vector<char>::const_iterator _current;
    std::vector<char>::const_iterator _end;
};

#endif
