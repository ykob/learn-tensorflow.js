const gulp = require('gulp');
const requireDir = require('require-dir');
const runSequence = require('run-sequence');
const browserSync = require('browser-sync');
const reload = browserSync.reload;

const $ = require('./gulp/plugins');
const DIR = require('./gulp/conf').DIR;

requireDir('./gulp/tasks');

gulp.task('predefault', cb => {
  runSequence(
    'cleanDest',
    ['pug', 'sass', 'watchify', 'copyToDest'],
    'serve',
    cb
  );
});

gulp.task('watch-sass', ['sass'], () => {
  reload();
});

gulp.task('default', ['predefault'], () => {
  $.watch(
    [`./${DIR.SRC}/**/*.{scss,sass}`],
    () => {
      gulp.start(['watch-sass'])
    }
  );

  $.watch(
    [`./${DIR.SRC}/**/*.pug`]
  ).on('change', reload);

  $.watch(
    [`./${DIR.DEST}/**/*.js`]
  ).on('change', reload);

  $.watch(
    [
      `./${DIR.SRC}/img/**/*.*`,
      `./${DIR.SRC}/font/**/*.*`,
      `./${DIR.SRC}/json/**/*.*`,
    ],
    () => {
      gulp.start(['copyToDest'])
    }
  ).on('change', reload);
});

gulp.task('build', cb => {
  runSequence(
    'cleanDest',
    ['pug', 'sass', 'browserify', 'copyToDest'],
    'cleanBuild',
    'replaceHtml',
    'cleanCss',
    'imagemin',
    'uglify',
    ['copyToBuild', 'copyPhpToBuild'],
    'sitemap',
    cb
  );
});

gulp.task('buildHtml', cb => {
  runSequence(
    'pug',
    'replaceHtml',
    cb
  );
});

gulp.task('buildCss', cb => {
  runSequence(
    'sass',
    'cleanCss',
    cb
  );
});

gulp.task('buildScript', cb => {
  runSequence(
    'browserify',
    'uglify',
    cb
  );
});
