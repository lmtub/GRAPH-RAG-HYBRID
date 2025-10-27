static void test_validate_fail_struct_missing(TestInputVisitorData *data,

                                              const void *unused)

{

    Error *err = NULL;

    Visitor *v;

    QObject *any;

    GenericAlternate *alt;

    bool present;

    int en;

    int64_t i64;

    uint32_t u32;

    int8_t i8;

    char *str;

    double dbl;



    v = validate_test_init(data, "{}");

    visit_start_struct(v, NULL, NULL, 0, &error_abort);

    visit_start_struct(v, "struct", NULL, 0, &err);

    error_free_or_abort(&err);

    visit_start_list(v, "list", NULL, 0, &err);

    error_free_or_abort(&err);

    visit_start_alternate(v, "alternate", &alt, sizeof(*alt), false, &err);

    error_free_or_abort(&err);

    visit_optional(v, "optional", &present);

    g_assert(!present);

    visit_type_enum(v, "enum", &en, EnumOne_lookup, &err);

    error_free_or_abort(&err);

    visit_type_int(v, "i64", &i64, &err);

    error_free_or_abort(&err);

    visit_type_uint32(v, "u32", &u32, &err);

    error_free_or_abort(&err);

    visit_type_int8(v, "i8", &i8, &err);

    error_free_or_abort(&err);

    visit_type_str(v, "i8", &str, &err);

    error_free_or_abort(&err);

    visit_type_number(v, "dbl", &dbl, &err);

    error_free_or_abort(&err);

    visit_type_any(v, "any", &any, &err);

    error_free_or_abort(&err);

    visit_type_null(v, "null", &err);

    error_free_or_abort(&err);

    visit_end_struct(v, NULL);

}
