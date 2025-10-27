static UserDefNested *nested_struct_create(void)

{

    UserDefNested *udnp = g_malloc0(sizeof(*udnp));

    udnp->string0 = strdup("test_string0");

    udnp->dict1.string1 = strdup("test_string1");

    udnp->dict1.dict2.userdef1 = g_malloc0(sizeof(UserDefOne));

    udnp->dict1.dict2.userdef1->base = g_new0(UserDefZero, 1);

    udnp->dict1.dict2.userdef1->base->integer = 42;

    udnp->dict1.dict2.userdef1->string = strdup("test_string");

    udnp->dict1.dict2.string2 = strdup("test_string2");

    udnp->dict1.has_dict3 = true;

    udnp->dict1.dict3.userdef2 = g_malloc0(sizeof(UserDefOne));

    udnp->dict1.dict3.userdef2->base = g_new0(UserDefZero, 1);

    udnp->dict1.dict3.userdef2->base->integer = 43;

    udnp->dict1.dict3.userdef2->string = strdup("test_string");

    udnp->dict1.dict3.string3 = strdup("test_string3");

    return udnp;

}
