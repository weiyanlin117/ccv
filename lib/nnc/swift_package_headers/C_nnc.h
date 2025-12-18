#ifndef C_nnc_umbrella_h
#define C_nnc_umbrella_h

// First, include the base ccv headers with relative paths 
// so the module importer knows where they are.
#import "../../ccv.h"
#import "../../ccv_internal.h"

// Now include the NNC headers. 
// When these files call #include "ccv.h", the compiler already 
// has the context from the imports above.
#import "../ccv_nnc.h"
#import "../ccv_nnc_easy.h"
#import "../ccv_nnc_internal.h"

#endif