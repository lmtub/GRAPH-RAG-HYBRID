void nonono(const char* file, int line, const char* msg) {

    fprintf(stderr, "Nonono! %s:%d %s\n", file, line, msg);

    exit(-5);

}
