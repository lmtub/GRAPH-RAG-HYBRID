static void init_types(void)

{

    static int inited;

    int i;



    if (inited) {

        return;

    }



    for (i = 0; i < MODULE_INIT_MAX; i++) {

        TAILQ_INIT(&init_type_list[i]);

    }



    inited = 1;

}
