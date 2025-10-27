static int vnc_set_x509_credential(VncDisplay *vs,

				   const char *certdir,

				   const char *filename,

				   char **cred,

				   int ignoreMissing)

{

    struct stat sb;



    if (*cred) {

	qemu_free(*cred);

	*cred = NULL;

    }



    *cred = qemu_malloc(strlen(certdir) + strlen(filename) + 2);



    strcpy(*cred, certdir);

    strcat(*cred, "/");

    strcat(*cred, filename);



    VNC_DEBUG("Check %s\n", *cred);

    if (stat(*cred, &sb) < 0) {

	qemu_free(*cred);

	*cred = NULL;

	if (ignoreMissing && errno == ENOENT)

	    return 0;

	return -1;

    }



    return 0;

}
