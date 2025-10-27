static inline void futex_wake(QemuEvent *ev, int n)

{


    if (n == 1) {

        pthread_cond_signal(&ev->cond);

    } else {

        pthread_cond_broadcast(&ev->cond);

    }


}