static ModuleTypeList *find_type(module_init_type type)

{

    ModuleTypeList *l;



    init_types();



    l = &init_type_list[type];



    return l;

}
