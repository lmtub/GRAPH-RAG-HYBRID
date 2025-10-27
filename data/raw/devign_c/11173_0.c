static void init_native_list(UserDefNativeListUnion *cvalue)

{

    int i;

    switch (cvalue->type) {

    case USER_DEF_NATIVE_LIST_UNION_KIND_INTEGER: {

        intList **list = &cvalue->u.integer.data;

        for (i = 0; i < 32; i++) {

            *list = g_new0(intList, 1);

            (*list)->value = i;

            (*list)->next = NULL;

            list = &(*list)->next;

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_S8: {

        int8List **list = &cvalue->u.s8.data;

        for (i = 0; i < 32; i++) {

            *list = g_new0(int8List, 1);

            (*list)->value = i;

            (*list)->next = NULL;

            list = &(*list)->next;

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_S16: {

        int16List **list = &cvalue->u.s16.data;

        for (i = 0; i < 32; i++) {

            *list = g_new0(int16List, 1);

            (*list)->value = i;

            (*list)->next = NULL;

            list = &(*list)->next;

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_S32: {

        int32List **list = &cvalue->u.s32.data;

        for (i = 0; i < 32; i++) {

            *list = g_new0(int32List, 1);

            (*list)->value = i;

            (*list)->next = NULL;

            list = &(*list)->next;

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_S64: {

        int64List **list = &cvalue->u.s64.data;

        for (i = 0; i < 32; i++) {

            *list = g_new0(int64List, 1);

            (*list)->value = i;

            (*list)->next = NULL;

            list = &(*list)->next;

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_U8: {

        uint8List **list = &cvalue->u.u8.data;

        for (i = 0; i < 32; i++) {

            *list = g_new0(uint8List, 1);

            (*list)->value = i;

            (*list)->next = NULL;

            list = &(*list)->next;

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_U16: {

        uint16List **list = &cvalue->u.u16.data;

        for (i = 0; i < 32; i++) {

            *list = g_new0(uint16List, 1);

            (*list)->value = i;

            (*list)->next = NULL;

            list = &(*list)->next;

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_U32: {

        uint32List **list = &cvalue->u.u32.data;

        for (i = 0; i < 32; i++) {

            *list = g_new0(uint32List, 1);

            (*list)->value = i;

            (*list)->next = NULL;

            list = &(*list)->next;

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_U64: {

        uint64List **list = &cvalue->u.u64.data;

        for (i = 0; i < 32; i++) {

            *list = g_new0(uint64List, 1);

            (*list)->value = i;

            (*list)->next = NULL;

            list = &(*list)->next;

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_BOOLEAN: {

        boolList **list = &cvalue->u.boolean.data;

        for (i = 0; i < 32; i++) {

            *list = g_new0(boolList, 1);

            (*list)->value = (i % 3 == 0);

            (*list)->next = NULL;

            list = &(*list)->next;

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_STRING: {

        strList **list = &cvalue->u.string.data;

        for (i = 0; i < 32; i++) {

            *list = g_new0(strList, 1);

            (*list)->value = g_strdup_printf("%d", i);

            (*list)->next = NULL;

            list = &(*list)->next;

        }

        break;

    }

    case USER_DEF_NATIVE_LIST_UNION_KIND_NUMBER: {

        numberList **list = &cvalue->u.number.data;

        for (i = 0; i < 32; i++) {

            *list = g_new0(numberList, 1);

            (*list)->value = (double)i / 3;

            (*list)->next = NULL;

            list = &(*list)->next;

        }

        break;

    }

    default:

        g_assert_not_reached();

    }

}
