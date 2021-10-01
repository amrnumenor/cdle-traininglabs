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
import org.nd4j.common.io.ClassPathResource;

import java.io.IOException;

import static org.bytedeco.opencv.global.opencv_core.BORDER_DEFAULT;
import static org.bytedeco.opencv.global.opencv_core.add;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.Sobel;

/*
 * TASKS:
 * -----
 * 1. Load and display x-ray.jpeg from the resources/image_processing folder
 * 2. Create and apply the vertical operator onto the input image
 * 3. Create and apply the horizontal operator onto the input image
 * 4. Apply both operator onto the input image (by adding them)
 * 5. Display the follwing:
 *       - image after applying vertical operator
 *       - image after applying horizontal operator
 *       - image after combining both operators
 *
 * */

public class SobelOperator {
    public static void main(String[] args) throws IOException {
        String path = new ClassPathResource("image_processing/x-ray.jpeg").getFile().getAbsolutePath();
        Mat src = imread(path);
        Display.display(src, "Original Image");

        Mat vertical = new Mat();
        Sobel(src, vertical, 0, 1, 0, 3, 1, 0, BORDER_DEFAULT);
        Display.display(vertical, "Sobel Vertical");

        Mat horizontal = new Mat();
        Sobel(src, horizontal, 0, 0, 1, 3, 1, 0, BORDER_DEFAULT);
        Display.display(horizontal, "Sobel Horizontal");

        Mat output = new Mat();
        add(vertical, horizontal, output);
        Display.display(output, "Final Output");
    }
}
