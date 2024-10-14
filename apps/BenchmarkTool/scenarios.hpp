/*
 *  Copyright (C):  2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2024/10/14 18:59:06
 *  Description:
 */
#include <Core/vector_db.hpp>

#include <map>

using namespace std;

void insertScenario(VectorDB &db);

void queryScenario(VectorDB &db);

void multiQueryInsertScenario(VectorDB &db);

extern map<string, function<void(VectorDB &db)>> scenarios;