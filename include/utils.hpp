//
//  utils.hpp
//  eigenValueSolver
//
//  Created by Adithya Vijaykumar on 17/09/2019.
//  Copyright © 2019 Adithya Vijaykumar. All rights reserved.
//

#ifndef utils_h
#define utils_h

#include "eigenValueSolver.hpp"

template <typename real>
void generateDDHermitianMatrix(El::DistMatrix<real> &A);
void initialiseMPI(El::Grid &);

#endif /* utils_h */
