static void test_native_list_integer_helper(TestInputVisitorData *data,

                                            const void *unused,

                                            UserDefNativeListUnionKind kind)

{

    UserDefNativeListUnion *cvalue = NULL;

    Visitor *v;

    GString *gstr_list = g_string_new("");

    GString *gstr_union = g_string_new("");

    int i;



    for (i = 0; i < 32; i++) {

        g_string_append_printf(gstr_list, "%d", i);

        if (i != 31) {

            g_string_append(gstr_list, ", ");

        }

    }

    g_string_append_printf(gstr_union,  "{ 'type': '%s', 'data': [ %s ] }",

                           UserDefNativeListUnionKind_lookup[kind],

                           gstr_list->str);

    v = visitor_input_test_init_raw(data,  gstr_union->str);



    visit_type_UserDefNativeListUnion(v, NULL, &cvalue, &error_abort);

    g_assert(cvalue != NULL);

    g_assert_cmpint(cvalue->type, ==, kind);



    switch (kind) {

    case USER_DEF_NATIVE_LIST_UNION_KIND_INTEGER: {

        intList *elem = NULL;

        for (i = 0, elem = cvalue->u.integer.data;

             elem; elem = elem->next, i++) {

            g_assert_cmpint(elem->value, ==, i);

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_S8: {

        int8List *elem = NULL;

        for (i = 0, elem = cvalue->u.s8.data; elem; elem = elem->next, i++) {

            g_assert_cmpint(elem->value, ==, i);

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_S16: {

        int16List *elem = NULL;

        for (i = 0, elem = cvalue->u.s16.data; elem; elem = elem->next, i++) {

            g_assert_cmpint(elem->value, ==, i);

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_S32: {

        int32List *elem = NULL;

        for (i = 0, elem = cvalue->u.s32.data; elem; elem = elem->next, i++) {

            g_assert_cmpint(elem->value, ==, i);

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_S64: {

        int64List *elem = NULL;

        for (i = 0, elem = cvalue->u.s64.data; elem; elem = elem->next, i++) {

            g_assert_cmpint(elem->value, ==, i);

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_U8: {

        uint8List *elem = NULL;

        for (i = 0, elem = cvalue->u.u8.data; elem; elem = elem->next, i++) {

            g_assert_cmpint(elem->value, ==, i);

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_U16: {

        uint16List *elem = NULL;

        for (i = 0, elem = cvalue->u.u16.data; elem; elem = elem->next, i++) {

            g_assert_cmpint(elem->value, ==, i);

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_U32: {

        uint32List *elem = NULL;

        for (i = 0, elem = cvalue->u.u32.data; elem; elem = elem->next, i++) {

            g_assert_cmpint(elem->value, ==, i);

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_U64: {

        uint64List *elem = NULL;

        for (i = 0, elem = cvalue->u.u64.data; elem; elem = elem->next, i++) {

            g_assert_cmpint(elem->value, ==, i);

        }

        break;

    }

    default:

        g_assert_not_reached();

    }



    g_string_free(gstr_union, true);

    g_string_free(gstr_list, true);

    qapi_free_UserDefNativeListUnion(cvalue);

}
