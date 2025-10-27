static void test_nested_struct(gconstpointer opaque)

{

    TestArgs *args = (TestArgs *) opaque;

    const SerializeOps *ops = args->ops;

    UserDefNested *udnp = nested_struct_create();

    UserDefNested *udnp_copy = NULL;

    Error *err = NULL;

    void *serialize_data;

    

    ops->serialize(udnp, &serialize_data, visit_nested_struct, &err);

    ops->deserialize((void **)&udnp_copy, serialize_data, visit_nested_struct, &err); 



    g_assert(err == NULL);

    nested_struct_compare(udnp, udnp_copy);



    nested_struct_cleanup(udnp);

    nested_struct_cleanup(udnp_copy);



    ops->cleanup(serialize_data);

    g_free(args);

}
