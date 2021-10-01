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
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/*
 *
 * 1. Go to https://image.online-convert.com/, convert resources/image_processing/opencv.png into the following format:
 *       - .bmp
 *       - .jpg
 *       - .tiff
 *     Save them to the same resources/image_processing folder.
 *
 *  2. Use the .imread function to load each all the images in resources/image_processing,
 *       and display them using Display.display
 *
 *
 *  3. Print the following image attributes:
 *       - depth
 *       - number of channel
 *       - width
 *       - height
 *
 *  4. Repeat step 2 & 3, but this time load the images in grayscale
 *
 *  5. Resize file
 *
 *  6. Write resized file to disk
 *
 * */

public class LoadImages {
    public static void main(String[] args) throws IOException {

        // get file path
        String path = new ClassPathResource("image_processing/sat_map3.jpg").getFile().getAbsolutePath();
        Mat inputImage = imread(path);

        // read graysclas
        Mat inputImageGS = imread(path, IMREAD_GRAYSCALE);

        // display image
        Display.display(inputImage, "Input Image");

        // image attributes
        System.out.println("Original Image");
        System.out.println("Width: " + inputImage.arrayWidth());
        System.out.println("Height: " + inputImage.arrayHeight());
        System.out.println("Channels: " + inputImage.arrayChannels());

        Mat downsized = new Mat();
        resize(inputImage, downsized, new Size(240, 240));

        // Display.display(downsized, "Downsized Image");

        System.out.println("\nDownsized Image");
        System.out.println("Width: " + downsized.arrayWidth());
        System.out.println("Height: " + downsized.arrayHeight());
        System.out.println("Channels: " + downsized.arrayChannels());

        Mat upsized_nn = new Mat();
        resize(downsized, upsized_nn, new Size(768, 506), 0, 0, INTER_NEAREST);

        Display.display(upsized_nn, "Upsized Image (NN)");

        System.out.println("\nUpsized Image");
        System.out.println("Width: " + upsized_nn.arrayWidth());
        System.out.println("Height: " + upsized_nn.arrayHeight());
        System.out.println("Channels: " + upsized_nn.arrayChannels());

        Mat upsized_ln = new Mat();
        Mat upsized_cn = new Mat();

        resize(downsized, upsized_ln, new Size(768, 506), 0, 0, INTER_LINEAR);
        resize(downsized, upsized_cn, new Size(768, 506), 0, 0, INTER_CUBIC);

        Display.display(upsized_ln, "Upsized Image (LINEAR)");
        Display.display(upsized_cn, "Upsized Image (CUBIC)");
    }
}
