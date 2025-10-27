static void test_endianness_combine(gconstpointer data)

{

    const TestCase *test = data;

    char *args;



    args = g_strdup_printf("-display none -M %s%s%s -device pc-testdev",

                           test->machine,

                           test->superio ? " -device " : "",

                           test->superio ?: "");

    qtest_start(args);

    isa_outl(test, 0xe0, 0x87654321);

    g_assert_cmphex(isa_inl(test, 0xe8), ==, 0x87654321);

    g_assert_cmphex(isa_inw(test, 0xea), ==, 0x8765);

    g_assert_cmphex(isa_inw(test, 0xe8), ==, 0x4321);



    isa_outw(test, 0xe2, 0x8866);

    g_assert_cmphex(isa_inl(test, 0xe8), ==, 0x88664321);

    g_assert_cmphex(isa_inw(test, 0xea), ==, 0x8866);

    g_assert_cmphex(isa_inw(test, 0xe8), ==, 0x4321);



    isa_outw(test, 0xe0, 0x4422);

    g_assert_cmphex(isa_inl(test, 0xe8), ==, 0x88664422);

    g_assert_cmphex(isa_inw(test, 0xea), ==, 0x8866);

    g_assert_cmphex(isa_inw(test, 0xe8), ==, 0x4422);



    isa_outb(test, 0xe3, 0x87);

    g_assert_cmphex(isa_inl(test, 0xe8), ==, 0x87664422);

    g_assert_cmphex(isa_inw(test, 0xea), ==, 0x8766);



    isa_outb(test, 0xe2, 0x65);

    g_assert_cmphex(isa_inl(test, 0xe8), ==, 0x87654422);

    g_assert_cmphex(isa_inw(test, 0xea), ==, 0x8765);

    g_assert_cmphex(isa_inw(test, 0xe8), ==, 0x4422);



    isa_outb(test, 0xe1, 0x43);

    g_assert_cmphex(isa_inl(test, 0xe8), ==, 0x87654322);

    g_assert_cmphex(isa_inw(test, 0xea), ==, 0x8765);

    g_assert_cmphex(isa_inw(test, 0xe8), ==, 0x4322);



    isa_outb(test, 0xe0, 0x21);

    g_assert_cmphex(isa_inl(test, 0xe8), ==, 0x87654321);

    g_assert_cmphex(isa_inw(test, 0xea), ==, 0x8765);

    g_assert_cmphex(isa_inw(test, 0xe8), ==, 0x4321);

    qtest_quit(global_qtest);

    g_free(args);

}
