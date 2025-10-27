static void sigbus_handler(int signal)

{

    siglongjmp(sigjump, 1);

}
