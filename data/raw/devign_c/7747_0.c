void test_self_modifying_code(void)

{

    int (*func)(void);



    func = (void *)code;

    printf("self modifying code:\n");

    printf("func1 = 0x%x\n", func());

    code[1] = 0x2;

    printf("func1 = 0x%x\n", func());

}
