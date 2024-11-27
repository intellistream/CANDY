//
// Created by zhonghao on 27/11/24.
//
#include "Algorithms/Manu/Log/WAL.hpp"

WAL::WAL(Type type, const std::string& vectorID, const std::vector<float>& vectorData,
         const std::string& label, float numericalField)
    : walType(type), vectorID(vectorID), vectorData(vectorData), label(label), numericalField(numericalField) {}
