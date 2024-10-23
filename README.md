<!-- Improved compatibility of back to top link: See: https://github.com/TechnicToms/lunar-technosignatures/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/TechnicToms/lunar-technosignatures">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="images/darkmode-overview.png">
        <img alt="Overview paper" src="images/lightmode-overview.png" width="65%">
    </picture>
  </a>

  <h3 align="center">Lunar technosignatures</h3>

  <p align="center">
    Repository for corresponding paper: Lunar Technosignatures: A Deep Learning Approach to Detectiong Apollo Landing Sites on the Lunar Surface
    <br />
    <a href="https://github.com/TechnicToms/lunar-technosignatures"><strong>Explore the results »</strong></a>
    <br />
    <br />
    <a href="https://github.com/TechnicToms/lunar-technosignatures">View Paper</a>
    ·
    <a href="https://https://github.com/TechnicToms/lunar-technosignatures/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/TechnicToms/lunar-technosignatures/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Uncovering anomalies on the lunar surface is crucial for understanding the Moon’s geological and astronomical history. By identifying and studying these anomalies, new theories about the changes that have occurred on the Moon can be developed or refined. This study seeks to enhance anomaly detection on the Moon and replace the time-consuming manual data search process by testing an anomaly detection method using theApollo landing sites. The landing sites are advantageous as they are both anomalous and can be located, enabling an assessment of the procedure. Our study compares the performance of various state-of-the-art machine learning algorithms in detecting anomalies in the Narrow-Angle Camera data from the Lunar Reconnaissance Orbiter spacecraft. The results demonstrate that our approach outperforms previous publications in accurately predicting landing site artifacts and technosignatures at the Apollo 15 and 17 landing sites. While our method achieves promising results, there is still room for improvement. Future refinements could focuson detecting more subtle anomalies, such as the rover tracks left by the Apollo missions.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

This project is built in Python and uses PyTorch and Pytorch lightning to train our multimodal model and tokenizers.

* [![Python-lang]][Python-url]
* [![Lightning-framework]][Lightning-url]
* [![PyTorch-framework]][PyTorch-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
5. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/TechnicToms/lunar-technosignatures/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Top contributors:

<a href="https://github.com/TechnicToms/lunar-technosignatures/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TechnicToms/lunar-technosignatures" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@ResearchGate](https://www.researchgate.net/profile/Tom-Sander-4) - tom.sander@tu-dortmund.de

Project Link: [https://github.com/TechnicToms/lunar-technosignatures](https://github.com/TechnicToms/lunar-technosignatures)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/TechnicToms/lunar-technosignatures.svg?style=for-the-badge
[contributors-url]: https://github.com/TechnicToms/lunar-technosignatures/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/TechnicToms/lunar-technosignatures.svg?style=for-the-badge
[forks-url]: https://github.com/TechnicToms/lunar-technosignatures/network/members
[stars-shield]: https://img.shields.io/github/stars/TechnicToms/lunar-technosignatures.svg?style=for-the-badge
[stars-url]: https://github.com/TechnicToms/lunar-technosignatures/stargazers
[issues-shield]: https://img.shields.io/github/issues/TechnicToms/lunar-technosignatures.svg?style=for-the-badge
[issues-url]: https://github.com/TechnicToms/lunar-technosignatures/issues
[license-shield]: https://img.shields.io/github/license/TechnicToms/lunar-technosignatures.svg?style=for-the-badge
[license-url]: https://github.com/TechnicToms/lunar-technosignatures/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/tom-sander-54374a20a/

[Python-lang]: https://img.shields.io/badge/Language-Python-2D618C?style=for-the-badge
[Python-url]: https://www.python.org/
[Lightning-framework]: https://img.shields.io/badge/Framework-Lightning-6019cb?style=for-the-badge
[Lightning-url]: https://lightning.ai/docs/pytorch/stable/
[PyTorch-framework]: https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=for-the-badge
[PyTorch-url]: https://pytorch.org/