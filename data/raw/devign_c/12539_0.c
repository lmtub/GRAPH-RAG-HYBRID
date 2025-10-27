static void test_dynamic_globalprop_subprocess(void)

{

    MyType *mt;

    static GlobalProperty props[] = {

        { TYPE_DYNAMIC_PROPS, "prop1", "101", true },

        { TYPE_DYNAMIC_PROPS, "prop2", "102", true },

        { TYPE_DYNAMIC_PROPS"-bad", "prop3", "103", true },

        /* .not_used=false to emulate what qdev_add_one_global() does: */

        { TYPE_UNUSED_HOTPLUG, "prop4", "104", false },

        { TYPE_UNUSED_NOHOTPLUG, "prop5", "105", true },

        { TYPE_NONDEVICE, "prop6", "106", true },

        {}

    };

    int all_used;



    qdev_prop_register_global_list(props);



    mt = DYNAMIC_TYPE(object_new(TYPE_DYNAMIC_PROPS));

    qdev_init_nofail(DEVICE(mt));



    g_assert_cmpuint(mt->prop1, ==, 101);

    g_assert_cmpuint(mt->prop2, ==, 102);

    all_used = qdev_prop_check_globals();

    g_assert_cmpuint(all_used, ==, 1);

    g_assert(!props[0].not_used);

    g_assert(!props[1].not_used);

    g_assert(props[2].not_used);

    g_assert(!props[3].not_used);

    g_assert(props[4].not_used);

    g_assert(props[5].not_used);

}
