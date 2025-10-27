Aml *init_aml_allocator(void)

{

    Aml *var;



    assert(!alloc_list);

    alloc_list = g_ptr_array_new();

    var = aml_alloc();

    return var;

}
