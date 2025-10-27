static void add_query_tests(QmpSchema *schema)

{

    SchemaInfoList *tail;

    SchemaInfo *si, *arg_type, *ret_type;

    const char *test_name;



    /* Test the query-like commands */

    for (tail = schema->list; tail; tail = tail->next) {

        si = tail->value;

        if (si->meta_type != SCHEMA_META_TYPE_COMMAND) {

            continue;

        }



        if (query_is_blacklisted(si->name)) {

            continue;

        }



        arg_type = qmp_schema_lookup(schema, si->u.command.arg_type);

        if (object_type_has_mandatory_members(arg_type)) {

            continue;

        }



        ret_type = qmp_schema_lookup(schema, si->u.command.ret_type);

        if (ret_type->meta_type == SCHEMA_META_TYPE_OBJECT

            && !ret_type->u.object.members) {

            continue;

        }



        test_name = g_strdup_printf("qmp/%s", si->name);

        qtest_add_data_func(test_name, si->name, test_query);

    }

}
