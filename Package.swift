// swift-tools-version: 6.0
import PackageDescription
import Foundation

let packageDir = URL(fileURLWithPath: #file).deletingLastPathComponent().path
#if os(Windows)
    let cuPath = ProcessInfo.processInfo.environment["CUDA_HOME"] ?? "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5"
    let cuIncludePath = "-I\(cuPath)\\include"
#elseif os(Linux)
    let cuPath = ProcessInfo.processInfo.environment["CUDA_HOME"] ?? "/usr/local/cuda"
    let cuIncludePath = "-I\(cuPath)/include"
#else
    fatalError("OS not supported \(os)")
#endif

let package = Package(
    name: "SwiftCU-example",
    dependencies: 
    [
        .package(url: "https://github.com/machineko/SwiftCU", branch: "main"),
        .package(url: "https://github.com/pvieito/PythonKit.git", branch: "master"),
    ],
    targets: [
        .target(
            name: "cxxCUBlas",
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath(cuIncludePath)
            ]
        ),
        .target(
            name: "SwiftCUBlas",
            dependencies: [
                "cxxCUBlas",
                .product(name: "SwiftCU", package: "SwiftCU"),
            ],
             swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(
                    ["-Xcc", cuIncludePath]
                )
            ]
        ),
          .testTarget(
            name: "SwiftCUBlasTests",
           
            dependencies: [
                "SwiftCU", "cxxCUBlas", "SwiftCUBlas",
                .product(name: "PythonKit", package: "PythonKit")
                // .product(name: "Testing", package: "swift-testing"), 
            ],
             swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        )
    ],
    cxxLanguageStandard: .cxx17
)