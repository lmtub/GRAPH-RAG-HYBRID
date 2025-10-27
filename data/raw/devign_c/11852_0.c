static void QEMU_NORETURN help(void)

{

    const char *help_msg =

           QEMU_IMG_VERSION

           "usage: qemu-img [standard options] command [command options]\n"

           "QEMU disk image utility\n"

           "\n"

           "    '-h', '--help'       display this help and exit\n"

           "    '-V', '--version'    output version information and exit\n"

           "    '-T', '--trace'      [[enable=]<pattern>][,events=<file>][,file=<file>]\n"

           "                         specify tracing options\n"

           "\n"

           "Command syntax:\n"

#define DEF(option, callback, arg_string)        \

           "  " arg_string "\n"

#include "qemu-img-cmds.h"

#undef DEF

#undef GEN_DOCS

           "\n"

           "Command parameters:\n"

           "  'filename' is a disk image filename\n"

           "  'objectdef' is a QEMU user creatable object definition. See the qemu(1)\n"

           "    manual page for a description of the object properties. The most common\n"

           "    object type is a 'secret', which is used to supply passwords and/or\n"

           "    encryption keys.\n"

           "  'fmt' is the disk image format. It is guessed automatically in most cases\n"

           "  'cache' is the cache mode used to write the output disk image, the valid\n"

           "    options are: 'none', 'writeback' (default, except for convert), 'writethrough',\n"

           "    'directsync' and 'unsafe' (default for convert)\n"

           "  'src_cache' is the cache mode used to read input disk images, the valid\n"

           "    options are the same as for the 'cache' option\n"

           "  'size' is the disk image size in bytes. Optional suffixes\n"

           "    'k' or 'K' (kilobyte, 1024), 'M' (megabyte, 1024k), 'G' (gigabyte, 1024M),\n"

           "    'T' (terabyte, 1024G), 'P' (petabyte, 1024T) and 'E' (exabyte, 1024P)  are\n"

           "    supported. 'b' is ignored.\n"

           "  'output_filename' is the destination disk image filename\n"

           "  'output_fmt' is the destination format\n"

           "  'options' is a comma separated list of format specific options in a\n"

           "    name=value format. Use -o ? for an overview of the options supported by the\n"

           "    used format\n"

           "  'snapshot_param' is param used for internal snapshot, format\n"

           "    is 'snapshot.id=[ID],snapshot.name=[NAME]', or\n"

           "    '[ID_OR_NAME]'\n"

           "  'snapshot_id_or_name' is deprecated, use 'snapshot_param'\n"

           "    instead\n"

           "  '-c' indicates that target image must be compressed (qcow format only)\n"

           "  '-u' enables unsafe rebasing. It is assumed that old and new backing file\n"

           "       match exactly. The image doesn't need a working backing file before\n"

           "       rebasing in this case (useful for renaming the backing file)\n"

           "  '-h' with or without a command shows this help and lists the supported formats\n"

           "  '-p' show progress of command (only certain commands)\n"

           "  '-q' use Quiet mode - do not print any output (except errors)\n"

           "  '-S' indicates the consecutive number of bytes (defaults to 4k) that must\n"

           "       contain only zeros for qemu-img to create a sparse image during\n"

           "       conversion. If the number of bytes is 0, the source will not be scanned for\n"

           "       unallocated or zero sectors, and the destination image will always be\n"

           "       fully allocated\n"

           "  '--output' takes the format in which the output must be done (human or json)\n"

           "  '-n' skips the target volume creation (useful if the volume is created\n"

           "       prior to running qemu-img)\n"

           "\n"

           "Parameters to check subcommand:\n"

           "  '-r' tries to repair any inconsistencies that are found during the check.\n"

           "       '-r leaks' repairs only cluster leaks, whereas '-r all' fixes all\n"

           "       kinds of errors, with a higher risk of choosing the wrong fix or\n"

           "       hiding corruption that has already occurred.\n"

           "\n"

           "Parameters to convert subcommand:\n"

           "  '-m' specifies how many coroutines work in parallel during the convert\n"

           "       process (defaults to 8)\n"

           "  '-W' allow to write to the target out of order rather than sequential\n"

           "\n"

           "Parameters to snapshot subcommand:\n"

           "  'snapshot' is the name of the snapshot to create, apply or delete\n"

           "  '-a' applies a snapshot (revert disk to saved state)\n"

           "  '-c' creates a snapshot\n"

           "  '-d' deletes a snapshot\n"

           "  '-l' lists all snapshots in the given image\n"

           "\n"

           "Parameters to compare subcommand:\n"

           "  '-f' first image format\n"

           "  '-F' second image format\n"

           "  '-s' run in Strict mode - fail on different image size or sector allocation\n"

           "\n"

           "Parameters to dd subcommand:\n"

           "  'bs=BYTES' read and write up to BYTES bytes at a time "

           "(default: 512)\n"

           "  'count=N' copy only N input blocks\n"

           "  'if=FILE' read from FILE\n"

           "  'of=FILE' write to FILE\n"

           "  'skip=N' skip N bs-sized blocks at the start of input\n";



    printf("%s\nSupported formats:", help_msg);

    bdrv_iterate_format(format_print, NULL);

    printf("\n");

    exit(EXIT_SUCCESS);

}
