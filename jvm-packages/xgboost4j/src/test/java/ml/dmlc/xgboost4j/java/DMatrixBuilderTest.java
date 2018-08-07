package ml.dmlc.xgboost4j.java;

import org.junit.Test;

import java.util.LinkedList;

import static org.junit.Assert.*;

/**
 * Created by: fitz
 * <p>
 * Date: 2018/3/5
 * <p>
 * Description:
 */
public class DMatrixBuilderTest {
    /*
    array([[1, 0, 2],
       [0, 0, 3],
       [0, 0, 0],
       [4, 5, 6]])
     */

    long[] headers = {0, 2, 3, 3, 6};
    int[] indices = {0, 2, 2, 0, 1, 2};
    float[] data = {1, 2, 3, 4, 5, 6};

    @Test
    public void testAddIndices(){
        DMatrixBuilder mb = new DMatrixBuilder(DMatrixBuilder.MATRIX_FORMAT.CSR)
                .addIndices(new int[]{0,2})
                .addIndices(new int[]{2})
                .addIndices(null)
                .addIndices(new int[]{0,1,2});

        LinkedList<Integer> expected = new LinkedList<>();
        expected.add(0);
        expected.add(2);
        expected.add(2);
        expected.add(0);
        expected.add(1);
        expected.add(2);
        assertEquals(expected.hashCode(),mb.getIndexList().hashCode());
    }

    @Test
    public void testAddDatas(){
        DMatrixBuilder mb = new DMatrixBuilder(DMatrixBuilder.MATRIX_FORMAT.CSR)
                .addDatas(new float[]{1f,2f})
                .addDatas(new float[]{3f})
                .addDatas(new float[]{4f,5f,6f});

        LinkedList<Float> expected = new LinkedList<>();
        expected.add(1f);
        expected.add(2f);
        expected.add(3f);
        expected.add(4f);
        expected.add(5f);
        expected.add(6f);

        assertEquals(expected, mb.getDataList());
    }

    @Test
    public void testAddHeaders(){

    }
}
