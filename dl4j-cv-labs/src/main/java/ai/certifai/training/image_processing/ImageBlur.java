/*
 * Copyright (c) 2020-2021 CertifAI Sdn. Bhd.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.certifai.training.image_processing;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.nd4j.common.io.ClassPathResource;

import java.io.IOException;

import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.GaussianBlur;
import static org.bytedeco.opencv.global.opencv_imgproc.medianBlur;

/*
 * Using code from LoadImages.java, read an image from the resources folder, and apply:
 *   1. Gaussian Blur
 *   2. Median Blur,
 *   and display both the original and blurred images.
 *   * You will need to create a new empty Mat object to store the blurred image.
 *   * Change the sigma value to observe the outcome of Gaussian blurring
 *
 * */


public class ImageBlur {
    public static void main(String[] args) throws IOException {
        String path = new ClassPathResource("image_processing/lena.png").getFile().getAbsolutePath();
        Mat inputImage= imread(path, IMREAD_GRAYSCALE);

        Mat blur_gaussian = new Mat();
        Mat blur_median = new Mat();

        GaussianBlur(inputImage, blur_gaussian, new Size(3, 3), 2.5); // v = sigma (std dev)
        medianBlur(inputImage, blur_median, 3); // i = filter size

        Display.display(blur_gaussian, "Gaussian Filter");
        Display.display(blur_gaussian, "Median Filter");
    }
}
