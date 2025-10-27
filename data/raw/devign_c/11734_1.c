static void convert_matrix(int *qmat, UINT16 *qmat16, const UINT16 *quant_matrix, int qscale)

{

    int i;



    if (av_fdct == jpeg_fdct_ifast) {

        for(i=0;i<64;i++) {

            /* 16 <= qscale * quant_matrix[i] <= 7905 */

            /* 19952         <= aanscales[i] * qscale * quant_matrix[i]           <= 249205026 */

            /* (1<<36)/19952 >= (1<<36)/(aanscales[i] * qscale * quant_matrix[i]) >= (1<<36)/249205026 */

            /* 3444240       >= (1<<36)/(aanscales[i] * qscale * quant_matrix[i]) >= 275 */

            

            qmat[block_permute_op(i)] = (int)((UINT64_C(1) << (QMAT_SHIFT + 11)) / 

                            (aanscales[i] * qscale * quant_matrix[block_permute_op(i)]));

        }

    } else {

        for(i=0;i<64;i++) {

            /* We can safely suppose that 16 <= quant_matrix[i] <= 255

               So 16           <= qscale * quant_matrix[i]             <= 7905

               so (1<<19) / 16 >= (1<<19) / (qscale * quant_matrix[i]) >= (1<<19) / 7905

               so 32768        >= (1<<19) / (qscale * quant_matrix[i]) >= 67

            */

            qmat[i]   = (1 << QMAT_SHIFT_MMX) / (qscale * quant_matrix[i]);

            qmat16[i] = (1 << QMAT_SHIFT_MMX) / (qscale * quant_matrix[block_permute_op(i)]);

        }

    }

}
