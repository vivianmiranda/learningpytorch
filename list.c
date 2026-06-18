// Schematically what is a Python list?
// PyListObject for a
//  ├── ob_size   = 3
//  ├── allocated = maybe >= 3
//  └── ob_item   = pointer to storage buffer
//                    |
//                    v
//              +-------------+-------------+-------------+
//              | PyObject*   | PyObject*   | PyObject*   |
//              +-------------+-------------+-------------+
//                    |             |             |
//                 PyLong(1)     PyLong(2)     PyLong(3)

typedef struct { // In CPython naming, the prefix ob means object.
  PyVarObject ob_base;  // the base object header ---------------
  PyObject**  ob_item;  // The object-item storage array --------
			// Why double pointer?
			// ob_item -> to 1st slot in the array
			// Each slot contains a PyObject*
  Py_ssize_t allocated; // capacity -----------------------------
} PyListObject;

// A Python object is a runtime value with:
// 1. identity: where this object lives in memory
// 2. type: what operations it supports
// 3. value: the data/content it represents
// Python list object
//  ├── identity: memory address of this list object
//  ├── type:     list
//  └── value:    references to 1, 2, 3
// Examples:
// PyListObject: Python list object
// PyLongObject: Python int object
// PyUnicodeObject: Python str object
// PyFloatObject: Python float object
typedef struct {
  Py_ssize_t ob_refcnt;   // reference count
  PyTypeObject *ob_type;  // points to &PyFloat_Type
  double ob_fval;
} PyFloatObject;



typedef struct { // Extension of PyObject that adds ob_size
    Py_ssize_t ob_refcnt;   // reference count
    PyTypeObject* ob_type;  // What is an object?
                            // Python's abstraction for data.
    Py_ssize_t ob_size;     // number of variable-size items
} PyVarObject;

// PyTypeObject: C struct that describes type of an object
// Examples:
// PyList_Type describes Python list
// PyLong_Type describes Python int
// PyUnicode_Type describes Python str
// PyFloat_Type describes Python float

// Summary: Python list has a lot more indirection than C arrays
//          That is bad for performance.
//	    That us why Pytorch has "tensors"
