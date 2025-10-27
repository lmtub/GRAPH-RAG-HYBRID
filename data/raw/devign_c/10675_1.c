static void visit_type_TestStruct(Visitor *v, TestStruct **obj,

                                  const char *name, Error **errp)

{

    Error *err = NULL;



    visit_start_struct(v, (void **)obj, "TestStruct", name, sizeof(TestStruct),

                       &err);

    if (err) {

        goto out;

    }



    visit_type_int(v, &(*obj)->integer, "integer", &err);

    visit_type_bool(v, &(*obj)->boolean, "boolean", &err);

    visit_type_str(v, &(*obj)->string, "string", &err);



    visit_end_struct(v, &err);



out:

    error_propagate(errp, err);

}
