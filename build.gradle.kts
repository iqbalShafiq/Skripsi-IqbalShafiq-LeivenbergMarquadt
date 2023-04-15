import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.6.10"
    application
}

group = "me.acer"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
    implementation(kotlin("reflect"))

    // multik
    implementation("org.jetbrains.kotlinx:multik-core:0.2.0")
    implementation("org.jetbrains.kotlinx:multik-default:0.2.1")

    //doyaaaaaken's kotlin-csv
    implementation("com.github.doyaaaaaken:kotlin-csv-jvm:1.6.0")

    //kotlin-grass
    implementation("io.github.blackmo18:kotlin-grass-core-jvm:1.0.0")
    implementation("io.github.blackmo18:kotlin-grass-parser-jvm:0.8.0")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.6.4")


    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")
}

tasks.test {
    useJUnitPlatform()
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}

tasks.withType<KotlinCompile>().configureEach {
    kotlinOptions {
        freeCompilerArgs = freeCompilerArgs + "-Xopt-in=kotlin.RequiresOptIn"
    }
}

application {
    mainClass.set("MainKt")
}