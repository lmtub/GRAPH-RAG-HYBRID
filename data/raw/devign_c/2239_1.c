static inline void qemu_assert(int cond, const char *msg)

{

    if (!cond) {

        fprintf (stderr, "badness: %s\n", msg);

        abort();

    }

}
