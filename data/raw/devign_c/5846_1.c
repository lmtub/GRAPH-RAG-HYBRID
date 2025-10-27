void help(void)

{

    printf("dct-test [-i] [<test-number>]\n"

           "test-number 0 -> test with random matrixes\n"

           "            1 -> test with random sparse matrixes\n"

           "            2 -> do 3. test from mpeg4 std\n"

           "-i          test IDCT implementations\n"

           "-4          test IDCT248 implementations\n");

    exit(1);

}
