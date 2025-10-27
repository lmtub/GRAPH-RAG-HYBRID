void module_call_init(module_init_type type)

{

    ModuleTypeList *l;

    ModuleEntry *e;



    l = find_type(type);



    TAILQ_FOREACH(e, l, node) {

        e->init();

    }

}
