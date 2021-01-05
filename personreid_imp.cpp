#include <stdlib.h>
#include <iostream>  

#include "python3.6m/Python.h" //Python.h中的P一定要大写，区分大小写

using namespace std;

int main(int argc, char* argv[])
{
    //初始化Python环境  
    Py_Initialize();

    PyRun_SimpleString("import sys");
    //用于导入调用pytorch第三方依赖
    PyRun_SimpleString("import torch");
    //添加Insert模块路径  
    //PyRun_SimpleString(chdir_cmd.c_str());
    PyRun_SimpleString("sys.path.append('./')");
    //PyRun_SimpleString("import mytest");

    //PyRun_SimpleString("mytest.Hello()");

    //导入模块  
    PyObject* pModule = PyImport_ImportModule("test_alignedreid");

    if (!pModule)
    {
        cout << "Python get module failed." << endl;
        return 0;
    }

    cout << "------Python get module succeed.------" << endl;

    //导入函数
    PyObject * pFunc = NULL;
    pFunc = PyObject_GetAttrString(pModule, "main");
    PyEval_CallObject(pFunc, NULL);

    Py_Finalize();

    system("pause");

    return 0;
}
