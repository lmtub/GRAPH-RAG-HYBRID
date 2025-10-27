static void test_io(void)

{

#ifndef _WIN32

/* socketpair(PF_UNIX) which does not exist on windows */



    int sv[2];

    int r;

    unsigned i, j, k, s, t;

    fd_set fds;

    unsigned niov;

    struct iovec *iov, *siov;

    unsigned char *buf;

    size_t sz;



    iov_random(&iov, &niov);

    sz = iov_size(iov, niov);

    buf = g_malloc(sz);

    for (i = 0; i < sz; ++i) {

        buf[i] = i & 255;

    }

    iov_from_buf(iov, niov, 0, buf, sz);



    siov = g_malloc(sizeof(*iov) * niov);

    memcpy(siov, iov, sizeof(*iov) * niov);



    if (socketpair(PF_UNIX, SOCK_STREAM, 0, sv) < 0) {

       perror("socketpair");

       exit(1);

    }



    FD_ZERO(&fds);



    t = 0;

    if (fork() == 0) {

       /* writer */



       close(sv[0]);

       FD_SET(sv[1], &fds);

       fcntl(sv[1], F_SETFL, O_RDWR|O_NONBLOCK);

       r = g_test_rand_int_range(sz / 2, sz);

       setsockopt(sv[1], SOL_SOCKET, SO_SNDBUF, &r, sizeof(r));



       for (i = 0; i <= sz; ++i) {

           for (j = i; j <= sz; ++j) {

               k = i;

               do {

                   s = g_test_rand_int_range(0, j - k + 1);

                   r = iov_send(sv[1], iov, niov, k, s);

                   g_assert(memcmp(iov, siov, sizeof(*iov)*niov) == 0);

                   if (r >= 0) {

                       k += r;

                       t += r;

                       usleep(g_test_rand_int_range(0, 30));

                   } else if (errno == EAGAIN) {

                       select(sv[1]+1, NULL, &fds, NULL, NULL);

                       continue;

                   } else {

                       perror("send");

                       exit(1);

                   }

               } while(k < j);

           }

       }




       exit(0);



    } else {

       /* reader & verifier */



       close(sv[1]);

       FD_SET(sv[0], &fds);

       fcntl(sv[0], F_SETFL, O_RDWR|O_NONBLOCK);

       r = g_test_rand_int_range(sz / 2, sz);

       setsockopt(sv[0], SOL_SOCKET, SO_RCVBUF, &r, sizeof(r));

       usleep(500000);



       for (i = 0; i <= sz; ++i) {

           for (j = i; j <= sz; ++j) {

               k = i;

               iov_memset(iov, niov, 0, 0xff, -1);

               do {

                   s = g_test_rand_int_range(0, j - k + 1);

                   r = iov_recv(sv[0], iov, niov, k, s);

                   g_assert(memcmp(iov, siov, sizeof(*iov)*niov) == 0);

                   if (r > 0) {

                       k += r;

                       t += r;

                   } else if (!r) {

                       if (s) {

                           break;

                       }

                   } else if (errno == EAGAIN) {

                       select(sv[0]+1, &fds, NULL, NULL, NULL);

                       continue;

                   } else {

                       perror("recv");

                       exit(1);

                   }

               } while(k < j);

               test_iov_bytes(iov, niov, i, j - i);

           }

        }






     }

#endif

}