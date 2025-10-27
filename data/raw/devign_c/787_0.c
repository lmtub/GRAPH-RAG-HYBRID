static void test_primitive_lists(gconstpointer opaque)

{

    TestArgs *args = (TestArgs *) opaque;

    const SerializeOps *ops = args->ops;

    PrimitiveType *pt = args->test_data;

    PrimitiveList pl = { .value = { NULL } };

    PrimitiveList pl_copy = { .value = { NULL } };

    PrimitiveList *pl_copy_ptr = &pl_copy;

    Error *err = NULL;

    void *serialize_data;

    void *cur_head = NULL;

    int i;



    pl.type = pl_copy.type = pt->type;



    /* build up our list of primitive types */

    for (i = 0; i < 32; i++) {

        switch (pl.type) {

        case PTYPE_STRING: {

            strList *tmp = g_new0(strList, 1);

            tmp->value = g_strdup(pt->value.string);

            if (pl.value.strings == NULL) {

                pl.value.strings = tmp;

            } else {

                tmp->next = pl.value.strings;

                pl.value.strings = tmp;

            }

            break;

        }

        case PTYPE_INTEGER: {

            intList *tmp = g_new0(intList, 1);

            tmp->value = pt->value.integer;

            if (pl.value.integers == NULL) {

                pl.value.integers = tmp;

            } else {

                tmp->next = pl.value.integers;

                pl.value.integers = tmp;

            }

            break;

        }

        case PTYPE_S8: {

            int8List *tmp = g_new0(int8List, 1);

            tmp->value = pt->value.s8;

            if (pl.value.s8_integers == NULL) {

                pl.value.s8_integers = tmp;

            } else {

                tmp->next = pl.value.s8_integers;

                pl.value.s8_integers = tmp;

            }

            break;

        }

        case PTYPE_S16: {

            int16List *tmp = g_new0(int16List, 1);

            tmp->value = pt->value.s16;

            if (pl.value.s16_integers == NULL) {

                pl.value.s16_integers = tmp;

            } else {

                tmp->next = pl.value.s16_integers;

                pl.value.s16_integers = tmp;

            }

            break;

        }

        case PTYPE_S32: {

            int32List *tmp = g_new0(int32List, 1);

            tmp->value = pt->value.s32;

            if (pl.value.s32_integers == NULL) {

                pl.value.s32_integers = tmp;

            } else {

                tmp->next = pl.value.s32_integers;

                pl.value.s32_integers = tmp;

            }

            break;

        }

        case PTYPE_S64: {

            int64List *tmp = g_new0(int64List, 1);

            tmp->value = pt->value.s64;

            if (pl.value.s64_integers == NULL) {

                pl.value.s64_integers = tmp;

            } else {

                tmp->next = pl.value.s64_integers;

                pl.value.s64_integers = tmp;

            }

            break;

        }

        case PTYPE_U8: {

            uint8List *tmp = g_new0(uint8List, 1);

            tmp->value = pt->value.u8;

            if (pl.value.u8_integers == NULL) {

                pl.value.u8_integers = tmp;

            } else {

                tmp->next = pl.value.u8_integers;

                pl.value.u8_integers = tmp;

            }

            break;

        }

        case PTYPE_U16: {

            uint16List *tmp = g_new0(uint16List, 1);

            tmp->value = pt->value.u16;

            if (pl.value.u16_integers == NULL) {

                pl.value.u16_integers = tmp;

            } else {

                tmp->next = pl.value.u16_integers;

                pl.value.u16_integers = tmp;

            }

            break;

        }

        case PTYPE_U32: {

            uint32List *tmp = g_new0(uint32List, 1);

            tmp->value = pt->value.u32;

            if (pl.value.u32_integers == NULL) {

                pl.value.u32_integers = tmp;

            } else {

                tmp->next = pl.value.u32_integers;

                pl.value.u32_integers = tmp;

            }

            break;

        }

        case PTYPE_U64: {

            uint64List *tmp = g_new0(uint64List, 1);

            tmp->value = pt->value.u64;

            if (pl.value.u64_integers == NULL) {

                pl.value.u64_integers = tmp;

            } else {

                tmp->next = pl.value.u64_integers;

                pl.value.u64_integers = tmp;

            }

            break;

        }

        case PTYPE_NUMBER: {

            numberList *tmp = g_new0(numberList, 1);

            tmp->value = pt->value.number;

            if (pl.value.numbers == NULL) {

                pl.value.numbers = tmp;

            } else {

                tmp->next = pl.value.numbers;

                pl.value.numbers = tmp;

            }

            break;

        }

        case PTYPE_BOOLEAN: {

            boolList *tmp = g_new0(boolList, 1);

            tmp->value = pt->value.boolean;

            if (pl.value.booleans == NULL) {

                pl.value.booleans = tmp;

            } else {

                tmp->next = pl.value.booleans;

                pl.value.booleans = tmp;

            }

            break;

        }

        default:

            g_assert_not_reached();

        }

    }



    ops->serialize((void **)&pl, &serialize_data, visit_primitive_list, &err);

    ops->deserialize((void **)&pl_copy_ptr, serialize_data, visit_primitive_list, &err);



    g_assert(err == NULL);

    i = 0;



    /* compare our deserialized list of primitives to the original */

    do {

        switch (pl_copy.type) {

        case PTYPE_STRING: {

            strList *ptr;

            if (cur_head) {

                ptr = cur_head;

                cur_head = ptr->next;

            } else {

                cur_head = ptr = pl_copy.value.strings;

            }

            g_assert_cmpstr(pt->value.string, ==, ptr->value);

            break;

        }

        case PTYPE_INTEGER: {

            intList *ptr;

            if (cur_head) {

                ptr = cur_head;

                cur_head = ptr->next;

            } else {

                cur_head = ptr = pl_copy.value.integers;

            }

            g_assert_cmpint(pt->value.integer, ==, ptr->value);

            break;

        }

        case PTYPE_S8: {

            int8List *ptr;

            if (cur_head) {

                ptr = cur_head;

                cur_head = ptr->next;

            } else {

                cur_head = ptr = pl_copy.value.s8_integers;

            }

            g_assert_cmpint(pt->value.s8, ==, ptr->value);

            break;

        }

        case PTYPE_S16: {

            int16List *ptr;

            if (cur_head) {

                ptr = cur_head;

                cur_head = ptr->next;

            } else {

                cur_head = ptr = pl_copy.value.s16_integers;

            }

            g_assert_cmpint(pt->value.s16, ==, ptr->value);

            break;

        }

        case PTYPE_S32: {

            int32List *ptr;

            if (cur_head) {

                ptr = cur_head;

                cur_head = ptr->next;

            } else {

                cur_head = ptr = pl_copy.value.s32_integers;

            }

            g_assert_cmpint(pt->value.s32, ==, ptr->value);

            break;

        }

        case PTYPE_S64: {

            int64List *ptr;

            if (cur_head) {

                ptr = cur_head;

                cur_head = ptr->next;

            } else {

                cur_head = ptr = pl_copy.value.s64_integers;

            }

            g_assert_cmpint(pt->value.s64, ==, ptr->value);

            break;

        }

        case PTYPE_U8: {

            uint8List *ptr;

            if (cur_head) {

                ptr = cur_head;

                cur_head = ptr->next;

            } else {

                cur_head = ptr = pl_copy.value.u8_integers;

            }

            g_assert_cmpint(pt->value.u8, ==, ptr->value);

            break;

        }

        case PTYPE_U16: {

            uint16List *ptr;

            if (cur_head) {

                ptr = cur_head;

                cur_head = ptr->next;

            } else {

                cur_head = ptr = pl_copy.value.u16_integers;

            }

            g_assert_cmpint(pt->value.u16, ==, ptr->value);

            break;

        }

        case PTYPE_U32: {

            uint32List *ptr;

            if (cur_head) {

                ptr = cur_head;

                cur_head = ptr->next;

            } else {

                cur_head = ptr = pl_copy.value.u32_integers;

            }

            g_assert_cmpint(pt->value.u32, ==, ptr->value);

            break;

        }

        case PTYPE_U64: {

            uint64List *ptr;

            if (cur_head) {

                ptr = cur_head;

                cur_head = ptr->next;

            } else {

                cur_head = ptr = pl_copy.value.u64_integers;

            }

            g_assert_cmpint(pt->value.u64, ==, ptr->value);

            break;

        }

        case PTYPE_NUMBER: {

            numberList *ptr;

            GString *double_expected = g_string_new("");

            GString *double_actual = g_string_new("");

            if (cur_head) {

                ptr = cur_head;

                cur_head = ptr->next;

            } else {

                cur_head = ptr = pl_copy.value.numbers;

            }

            /* we serialize with %f for our reference visitors, so rather than

             * fuzzy floating math to test "equality", just compare the

             * formatted values

             */

            g_string_printf(double_expected, "%.6f", pt->value.number);

            g_string_printf(double_actual, "%.6f", ptr->value);

            g_assert_cmpstr(double_actual->str, ==, double_expected->str);

            g_string_free(double_expected, true);

            g_string_free(double_actual, true);

            break;

        }

        case PTYPE_BOOLEAN: {

            boolList *ptr;

            if (cur_head) {

                ptr = cur_head;

                cur_head = ptr->next;

            } else {

                cur_head = ptr = pl_copy.value.booleans;

            }

            g_assert_cmpint(!!pt->value.boolean, ==, !!ptr->value);

            break;

        }

        default:

            g_assert_not_reached();

        }

        i++;

    } while (cur_head);



    g_assert_cmpint(i, ==, 33);



    ops->cleanup(serialize_data);

    dealloc_helper(&pl, visit_primitive_list, &err);

    g_assert(!err);

    dealloc_helper(&pl_copy, visit_primitive_list, &err);

    g_assert(!err);

    g_free(args);

}
