static void nested_struct_compare(UserDefTwo *udnp1, UserDefTwo *udnp2)

{

    g_assert(udnp1);

    g_assert(udnp2);

    g_assert_cmpstr(udnp1->string0, ==, udnp2->string0);

    g_assert_cmpstr(udnp1->dict1->string1, ==, udnp2->dict1->string1);

    g_assert_cmpint(udnp1->dict1->dict2->userdef->base->integer, ==,

                    udnp2->dict1->dict2->userdef->base->integer);

    g_assert_cmpstr(udnp1->dict1->dict2->userdef->string, ==,

                    udnp2->dict1->dict2->userdef->string);

    g_assert_cmpstr(udnp1->dict1->dict2->string, ==,

                    udnp2->dict1->dict2->string);

    g_assert(udnp1->dict1->has_dict3 == udnp2->dict1->has_dict3);

    g_assert_cmpint(udnp1->dict1->dict3->userdef->base->integer, ==,

                    udnp2->dict1->dict3->userdef->base->integer);

    g_assert_cmpstr(udnp1->dict1->dict3->userdef->string, ==,

                    udnp2->dict1->dict3->userdef->string);

    g_assert_cmpstr(udnp1->dict1->dict3->string, ==,

                    udnp2->dict1->dict3->string);

}
