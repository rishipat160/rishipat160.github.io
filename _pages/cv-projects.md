---
title: "CV Projects"
permalink: /cv-projects/
layout: splash
author_profile: false
classes: wide
---

<h1 class="page-title">Computer Vision Projects</h1>

<div class="projects-container">

<div class="project-card" id="object-recognition-cpp">
  <h2>Object Recognition with C++: Real-time Detection System</h2>
  
  <div class="project-metadata">
    <span class="project-tech"><i class="fab fa-cuttlefish"></i> C++, OpenCV, Standard Library</span>
    <a href="https://github.com/rishipat160/CppObjectRecognition" class="project-link"><i class="fab fa-github"></i> Code</a>
    <a href="https://www.youtube.com/watch?v=UCClHovymdE" class="project-link"><i class="fas fa-play-circle"></i> Demo</a>
    <a href="/assets/files/object_recognition_paper.pdf" class="project-link"><i class="fas fa-file-pdf"></i> Paper</a>
  </div>

  <div class="project-summary">
    <ul>
      <li>Built a real-time object recognition system using classical computer vision techniques (no deep learning)</li>
      <li>Implemented feature extraction using shape descriptors, Hu moments, and geometric properties</li>
      <li>Created a classification system with multiple distance metrics achieving >90% accuracy for distinct objects</li>
    </ul>
  </div>

  <details>
    <summary><strong>Project Overview</strong></summary>
    <div class="project-details">
      <p>This project implements a real-time object recognition system using C++ and OpenCV. The system can detect, track, and classify common objects using computer vision techniques and feature-based classification. Unlike deep learning approaches, this system relies on classical computer vision algorithms and geometric feature extraction, making it lightweight and suitable for embedded systems with limited computational resources.</p>
      
      <p>The system works by applying adaptive thresholding to isolate objects from the background, followed by connected component analysis to identify distinct regions. The adaptive thresholding dynamically adjusts to lighting conditions, making the system robust to varying illumination. Morphological operations (erosion and dilation) are then applied to reduce noise and improve region coherence.</p>
      
      <h3>Feature Extraction</h3>
      <p>For each detected region, the system extracts a set of shape-based features including:</p>
      
      <ul>
        <li><strong>Percent filled</strong> (area / bounding box area): Measures how "solid" an object is</li>
        <li><strong>Aspect ratio</strong>: Width-to-height ratio of the bounding box</li>
        <li><strong>Hu moments</strong>: Specifically the first two Hu moments (rotation invariant shape descriptors)</li>
        <li><strong>Orientation and principal axis</strong>: Direction of maximum variance in the object</li>
      </ul>
      
      <p>The feature extraction is implemented in C++ as shown in this code snippet:</p>
      
      {% highlight cpp %}
RegionFeatures computeRegionFeatures(const cv::Mat& labelsMat, int regionId) {
    cv::Mat regionMask = (labelsMat == regionId);
    
    cv::Moments m = cv::moments(regionMask, true);
    
    RegionFeatures features;
    
    features.center = cv::Point2f(m.m10/m.m00, m.m01/m.m00);
    features.orientation = 0.5 * atan2(2*m.mu11, m.mu20 - m.mu02);
    
    double theta = features.orientation;
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);
    
    // Calculate oriented bounding box dimensions
    double minAlongAxis = DBL_MAX, maxAlongAxis = -DBL_MAX;
    double minPerpAxis = DBL_MAX, maxPerpAxis = -DBL_MAX;
    
    for(int y = 0; y < labelsMat.rows; y++) {
        for(int x = 0; x < labelsMat.cols; x++) {
            if(labelsMat.at<int>(y, x) == regionId) {
                double alongAxis = (x - features.center.x) * cosTheta + 
                                  (y - features.center.y) * sinTheta;
                double perpAxis = -(x - features.center.x) * sinTheta + 
                                  (y - features.center.y) * cosTheta;
                
                minAlongAxis = std::min(minAlongAxis, alongAxis);
                maxAlongAxis = std::max(maxAlongAxis, alongAxis);
                minPerpAxis = std::min(minPerpAxis, perpAxis);
                maxPerpAxis = std::max(maxPerpAxis, perpAxis);
            }
        }
    }
    
    double width = maxAlongAxis - minAlongAxis;
    double height = maxPerpAxis - minPerpAxis;
    
    features.orientedBox = cv::RotatedRect(
        features.center, 
        cv::Size2f(width, height), 
        theta * 180.0 / CV_PI);
    
    features.percentFilled = m.m00 / (width * height);
    features.aspectRatio = width > height ? width / height : height / width;
    
    double huMoments[7];
    cv::HuMoments(m, huMoments);
    features.hu1 = -std::log10(std::abs(huMoments[0]));
    features.hu2 = -std::log10(std::abs(huMoments[1]));
    
    return features;
}
      {% endhighlight %}
      
      <h3>Classification System</h3>
      <p>These features are normalized to ensure each contributes equally to the classification process. The system maintains a database of known objects with their corresponding feature vectors. During recognition, incoming object features are compared against this database using various distance metrics:</p>
      
      <ul>
        <li><strong>Euclidean distance</strong>: Standard geometric distance in feature space</li>
        <li><strong>Scaled Euclidean</strong>: Weighted distance giving more importance to discriminative features</li>
        <li><strong>Cosine similarity</strong>: Measures the angle between feature vectors</li>
        <li><strong>Scaled L1 (Manhattan)</strong>: Sum of absolute differences with feature weighting</li>
      </ul>
      
      <p>The classification algorithm is implemented as follows:</p>
      
      {% highlight cpp %}
std::pair<std::string, double> classifyObjectWithConfidence(const RegionFeatures& features, int distanceMetric) {
    std::vector<DatabaseEntry> database = loadDatabase("data/object_features.csv");
    
    if (database.empty()) {
        return std::make_pair("Unknown (no database)", 0.0);
    }
    
    // Calculate standard deviations for normalization
    std::vector<double> stdDevs = calculateStdDevs(database);
    
    // Find nearest neighbor
    std::string bestMatch = "Unknown";
    double minDistance = DBL_MAX;
    
    for (const auto& entry : database) {
        double dist = 0.0;
        
        switch(distanceMetric) {
            case 0: // Simple Euclidean distance
                dist = sqrt(
                    pow(features.percentFilled - entry.features[0], 2) +
                    pow(features.aspectRatio - entry.features[1], 2) +
                    pow(features.hu1 - entry.features[2], 2) +
                    pow(features.hu2 - entry.features[3], 2)
                );
                break;
                
            case 1: // Scaled Euclidean distance (normalized by std dev)
                dist = sqrt(
                    pow((features.percentFilled - entry.features[0]) / stdDevs[0], 2) +
                    pow((features.aspectRatio - entry.features[1]) / stdDevs[1], 2) +
                    pow((features.hu1 - entry.features[2]) / stdDevs[2], 2) +
                    pow((features.hu2 - entry.features[3]) / stdDevs[3], 2)
                );
                break;
                
            case 2: // Cosine distance (1 - cosine similarity)
                {
                    double dotProduct = 
                        features.percentFilled * entry.features[0] +
                        features.aspectRatio * entry.features[1] +
                        features.hu1 * entry.features[2] +
                        features.hu2 * entry.features[3];
                        
                    double norm1 = sqrt(
                        pow(features.percentFilled, 2) +
                        pow(features.aspectRatio, 2) +
                        pow(features.hu1, 2) +
                        pow(features.hu2, 2)
                    );
                    
                    double norm2 = sqrt(
                        pow(entry.features[0], 2) +
                        pow(entry.features[1], 2) +
                        pow(entry.features[2], 2) +
                        pow(entry.features[3], 2)
                    );
                    
                    double similarity = dotProduct / (norm1 * norm2);
                    dist = 1.0 - similarity; 
                }
                break;
                
            case 3: // Scaled L1 (Manhattan) distance
                dist = 
                    fabs(features.percentFilled - entry.features[0]) / stdDevs[0] +
                    fabs(features.aspectRatio - entry.features[1]) / stdDevs[1] +
                    fabs(features.hu1 - entry.features[2]) / stdDevs[2] +
                    fabs(features.hu2 - entry.features[3]) / stdDevs[3];
                break;
        }
        
        if (dist < minDistance) {
            minDistance = dist;
            bestMatch = entry.label;
        }
    }
    
    double confidence = 0.0;
    double threshold = (distanceMetric == 2) ? 0.5 : 5.0; 
    
    if (minDistance < threshold) {
        confidence = 100.0 * (1.0 - minDistance/threshold);
        confidence = std::max(0.0, std::min(100.0, confidence)); // Clamp to 0-100%
    }
    
    // If confidence is too low, return Unknown
    if (confidence < 50.0) {
        return std::make_pair("Unknown", confidence);
    }
    
    return std::make_pair(bestMatch, confidence);
}
      {% endhighlight %}
      
      <h3>Evaluation System</h3>
      <p>The system includes an evaluation mode that generates a confusion matrix to assess classification accuracy. This allows for quantitative performance analysis across different objects and distance metrics:</p>
      
      {% highlight cpp %}
// Confusion matrix generation
case 't': { // test current object and update confusion matrix
    if (evaluationMode && !currentTrueLabel.empty()) {
        if (g_labels.empty()) {
            std::cout << "No valid regions found. Try again." << std::endl;
            break;
        }
        
        // Find first valid region
        bool foundValidRegion = false;
        int objectRegion = 0;
        
        for (int i = 0; i < g_labels.rows && !foundValidRegion; i++) {
            for (int j = 0; j < g_labels.cols && !foundValidRegion; j++) {
                if (g_labels.at<int>(i, j) > 0) {
                    objectRegion = g_labels.at<int>(i, j);
                    foundValidRegion = true;
                }
            }
        }
        
        // Compute features and classify
        RegionFeatures features = computeRegionFeatures(g_labels, objectRegion);
        auto classification = classifyObjectWithConfidence(features);
        std::string predictedLabel = classification.first;
        
        // Find indices for the confusion matrix
        int trueIndex = -1, predIndex = -1;
        for (int i = 0; i < objectLabels.size(); i++) {
            if (objectLabels[i] == currentTrueLabel) trueIndex = i;
            if (objectLabels[i] == predictedLabel) predIndex = i;
        }
        
        if (trueIndex >= 0 && predIndex >= 0) {
            confusionMatrix[trueIndex][predIndex]++;
            std::cout << "Recorded: True=" << currentTrueLabel 
                      << ", Predicted=" << predictedLabel << std::endl;
        }
    }
    break;
}
      {% endhighlight %}
      
      <h3>Interactive Training Mode</h3>
      <p>The system includes an interactive training mode that allows users to add new objects to the database. This is implemented through a simple interface that captures feature vectors for new objects:</p>
      
      {% highlight cpp %}
void saveFeatureVector(const RegionFeatures& features, const std::string& label) {
    std::ofstream file("data/object_features.csv", std::ios::app); 
    if (!file.is_open()) {
        std::cerr << "Error: Could not open database file." << std::endl;
        return;
    }
    
    // Save the feature vector with its label
    file << label << ","
         << features.percentFilled << ","
         << features.aspectRatio << ","
         << features.hu1 << ","
         << features.hu2 << "\n";
    
    std::cout << "Saved feature vector for object: " << label << std::endl;
    file.close();
}
      {% endhighlight %}
      
      <h3>System Architecture</h3>
      <p>The project is organized with a modular architecture:</p>
      <ul>
        <li><strong>threshold.hpp/cpp</strong>: Core image processing and feature extraction</li>
        <li><strong>vidDisplay.cpp</strong>: Main application loop and user interface</li>
        <li><strong>Makefile</strong>: Build system for compiling the application</li>
      </ul>
      
      <p>The build system is configured using a simple Makefile:</p>
      
      {% highlight makefile %}
CC = cl
CFLAGS = /MD /EHsc
INCLUDES = /I "include" /I "..\opencv\build\include"
LIBPATH = /link /LIBPATH:"..\opencv\build\x64\vc16\lib" 
LIBS = opencv_world4110.lib 
SRCDIR = src

vidDisplay:
	$(CC) $(CFLAGS) $(INCLUDES) $(SRCDIR)/vidDisplay.cpp $(SRCDIR)/threshold.cpp /Fobin/ /Febin/$@ $(LIBPATH) $(LIBS)

runVid: vidDisplay
	.\bin\vidDisplay.exe

clean:
	del bin\*.obj bin\*.exe *.jpg 
      {% endhighlight %}
      
      <h3>Results and Performance</h3>
      <p>The system achieves high accuracy for a limited set of objects (>90% for distinct objects) and operates in real-time on standard hardware. The performance varies based on:</p>
      <ul>
        <li>Object distinctiveness (shape differences)</li>
        <li>Lighting conditions</li>
        <li>Background complexity</li>
        <li>Choice of distance metric</li>
      </ul>
      
      <div class="project-image">
        <iframe width="100%" height="50" src="https://www.youtube.com/embed/UCClHovymdE" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        <p class="caption">Demo video of the object recognition system in action</p>
      </div>
      
      <h3>Limitations and Future Improvements</h3>
      <p>The most significant challenge encountered was the impact of lighting on training and recognition. Objects trained under bright lighting would often be misclassified under dim lighting due to changes in the thresholded region shapes. This highlights the importance of training with multiple lighting conditions.</p>
      
      <p>Future improvements could include:</p>
      <ul>
        <li>Better handling of lighting variations through adaptive preprocessing</li>
        <li>Adding more shape features to improve discrimination between similar objects</li>
        <li>Implementing a more robust classification algorithm</li>
        <li>Creating a more structured database for storing object features</li>
        <li>Improving the user interface for the training mode</li>
      </ul>
      
      <p>Below is a link to the paper I wrote for this project highlighting how it was built and the results.</p>
      
      <div class="pdf-container">
        <iframe src="/assets/files/object_recognition_paper.pdf" width="100%" height="500px"></iframe>
      </div>
      
      <p class="disclaimer"><em>Note: This project was developed as a proof-of-concept with a focus on functionality and results. The codebase would benefit from refactoring to improve modularity, readability, and maintainability. Future iterations would separate components into distinct modules and implement better error handling.</em></p>
    </div>
  </details>
</div>

<div class="project-card" id="realtime-filters">
  <h2>Real-time Filters Application with OpenCV: Interactive Image Processing</h2>
  
  <div class="project-metadata">
    <span class="project-tech"><i class="fab fa-cuttlefish"></i> C++, OpenCV, Image Processing</span>
    <span class="project-status"><i class="fas fa-clock"></i> Coming 03/10/2025</span>
  </div>

  <div class="project-summary">
    <ul>
      <li>Developed a real-time video filtering application with interactive controls</li>
      <li>Implemented various image processing algorithms for visual effects</li>
      <li>Created a user-friendly interface for combining and customizing filters</li>
    </ul>
  </div>

  <details>
    <summary><strong>Project Overview</strong></summary>
    <div class="project-details">
      <p>This project implements a real-time video filtering application using OpenCV and C++. The system will apply various image processing filters to webcam input in real-time, allowing users to interactively modify and combine different effects.</p>
    </div>
  </details>
</div>

</div>